import time
import os
import numpy as np
from collections import Counter
import cv2
import torch
from torch.nn.functional import softmax

import gesture_lib.ops.keypoint as K
from gesture_lib.datasets import build_dataset
from gesture_lib.datasets import Body3DExtractor
from gesture_lib.models import build_model, build_matcher
from gesture_lib.ops.box import box_iou
from gesture_lib.ops.yaml_utils import parse_yaml
from gesture_lib.ops.realsense import RsRGBD
from gesture_lib.models import MemoryManager


class InferenceRGBD(object):
    '''inference of body keypoints model for video from realsense
    TODO:
        1. gesture recognition is not stable, need better smooth strategy
           or model with stronger robustness (more data, better model)
        2. try multi-thread for higher inference speed
        3. is siamese suitable for tracking in this application?
    '''

    def __init__(self, cfg_path, checkpoints):
        assert os.path.splitext(checkpoints)[-1] == '.pth'

        cfg = parse_yaml(cfg_path)
        print(dict(cfg))
        core_cfg = parse_yaml("gesture_lib/configs.yaml")

        # configure dataset
        data_cfg = cfg.dataset
        self.cls_names = data_cfg.test.cls_names
        self.body_names = data_cfg.test.body_names
        self.dataset = build_dataset(data_cfg.test)
        self.idx_list = self.dataset.idx_list
        self.transforms = self.dataset.transforms

        # configure realsense camera
        self.D435 = core_cfg.camera_id.D435
        self.rs_camera = RsRGBD(serial_number=self.D435)
        self.width = core_cfg.width
        self.height = core_cfg.height

        # configure pose estimation
        pose_params = core_cfg.pose_params
        self.mode = pose_params.mode
        self.pose_w = pose_params.pose_w
        self.pose_h = pose_params.pose_h
        self.extractor = Body3DExtractor()

        # configure gesture recognition
        self.max_person_one_frame = core_cfg.max_person_one_frame
        self.seq_len = core_cfg.gesture_params.seq_len
        self.score_thr = core_cfg.gesture_params.score_thr
        self._configure_gesture_rec(cfg.model, checkpoints)

        # cache for output (for smoothing the noise)
        self.output_cache = {}
        self.output_cache_params = core_cfg.output_cache_params

        # person info
        self.person_ids = []  # for tracking: [id, sim]
        self.target = {'person_id': '0', 'miss_times': 0, 'vote_prop': []}
        self.target_params = core_cfg.target_params

        # configure matcher and memory for tracking
        matcher_params = core_cfg.matcher_params
        matcher_params.mode = core_cfg.pose_params.mode
        self.matcher = build_matcher(matcher_params)
        memory_params = core_cfg.memory_params
        memory_params.matcher = self.matcher
        self.memory_manager = MemoryManager(**memory_params)
        self.sim_thr1 = memory_params.sim_thr1
        self.sim_thr2 = memory_params.sim_thr2

    def _configure_gesture_rec(self, model_cfg, checkpoints):
        # result of current frame
        self.points_seq = {}  # input sequence for gesture model
        self.predict_result = {}  # gesture model output

        # init params for smoothing result
        self.predict = {}
        self.smooth_count = 0

        # configure model
        self.gesture_model = build_model(model_cfg).cuda()
        self.gesture_model.load_state_dict(torch.load(checkpoints))
        self.gesture_model.eval()

    def _get_aligned_frame(self, frames):
        return self.extractor._get_aligned_frame(frames)

    def _body_keypoints(self, img):
        '''detect body keypoints from a frame

        Args:
            frame: one frame from realsense, including color
                and depth frame info

        Return:
            keypoints: [ndarray], pose keypoints array,
                shape (N, keypoint_num, :), (x, y) or (x, y, score)
            cv_output: [ndarray]: output image with keypoints info.

        '''
        return self.extractor.body_keypoints(img)

    def _keypoint_to_point(self, keypoint, depth_frame):
        point = self.extractor.keypoint_to_point(keypoint, depth_frame)
        return point

    def _post_process(self, points_array, label):
        if label == "wave":
            return K.post_process_wave(points_array, self.mode)
        elif label == "come":
            return K.post_process_come(points_array, self.mode)
        elif label == "hello":
            return K.post_process_hello(points_array, self.mode)
        else:
            return True

    def _gesture_rec(self, person_id, point_array):
        predict_label = "others"
        score = 0.0
        if K.is_static(point_array):
            print('static state')
            return predict_label, score
        point_tensor = self.transforms(point_array)
        point_tensor = torch.Tensor(point_tensor).unsqueeze(0).cuda()
        with torch.no_grad():
            predict = self.gesture_model(point_tensor)[0].cpu()
            predict = softmax(predict, dim=0)
            predict = self._smooth_predict(person_id, predict)

        idx = np.argmax(predict)
        score = float(predict[idx])
        if score > self.score_thr:
            predict_label = self.cls_names[idx]
            flag = self._post_process(point_array, predict_label)
            predict_label = predict_label if flag is True else "others"
        else:
            print("ignore low score results: ", np.round(predict, 3))
        return predict_label, score

    def _smooth_predict(self, person_id, predict, alpha=0.5, clean_num=100):
        if self.smooth_count > clean_num:
            self.predict = {}
            self.smooth_count = 0
        if person_id not in self.predict.keys():
            self.predict[person_id] = predict
            self.smooth_count += 1
            return predict
        predict_s = (1-alpha) * self.predict[person_id] + alpha * predict
        self.predict[person_id] = predict_s
        self.smooth_count += 1
        return predict_s

    def _update_target_vote_prop(self, label, default='others'):
        prop_num = len(self.target['vote_prop'])
        if prop_num >= self.target_params['vote_len']:
            self.target['vote_prop'].pop(0)
        self.target['vote_prop'].append(label)
        vote_prop = self.target['vote_prop']
        if len(vote_prop) < self.target_params['vote_len']:
            return default
        else:
            prop_label = Counter(vote_prop).most_common(1)[0][0]
            return prop_label

    def _draw_person_ids(self, cv_output, skeletons):
        '''
        Args:
            keypoints: shape (N, 25, 3) [x, y, score]
            person_ids: list, len = N
        '''
        result_img = cv_output.copy()
        print("self.person_ids: ", self.person_ids)
        for idx, (person_id, sim) in enumerate(self.person_ids):
            sim = round(sim, 2)
            keypoint = skeletons[idx][0]
            keypoint = keypoint[keypoint[:, 0] > 0]
            x1 = int(np.min(keypoint[:, 0]))
            y1 = int(np.min(keypoint[:, 1]))
            x2 = int(np.max(keypoint[:, 0]))
            y2 = int(np.max(keypoint[:, 1]))
            cv2.rectangle(result_img, (x1, y1), (x2, y2), color=(255, 0, 0),
                          thickness=2)
            cv2.putText(result_img, person_id + ' '+str(sim), (x1, y1+20),
                        cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
            if person_id in self.predict_result.keys():
                predict_label = self.predict_result[person_id][0]
                score = np.round(self.predict_result[person_id][1], 2)
                if predict_label == "others":
                    continue
                ss = predict_label + ' ' + str(score)
                cv2.putText(result_img, ss, (x1, y1+40),
                            cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 0, 255), 2)
        return result_img

    def _draw_fps(self, image, t):
        image = np.array(image, dtype=np.uint8)
        speed = str(int(1 / t)) + ' FPS'
        cv2.putText(image, speed, (image.shape[1]-120, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
        return image

    def _draw_target(self, cv_output):
        result_img = cv_output.copy()
        target_person = self.target['person_id']
        if target_person == '0':
            return result_img

        target_box = self.memory_manager.memory[target_person]['keypoint_box']
        x1, y1, x2, y2 = np.int32(target_box)
        cv2.rectangle(result_img, (x1, y1), (x2, y2), color=(0, 255, 0),
                      thickness=4)
        return result_img

    def _is_valid_body_angle(self, angle):
        # print("angle = {}".format(angle))
        if angle[0] * angle[1] * angle[2] == 0:
            return False
        if angle[0] < 20 or angle[0] > 160:
            return False
        if angle[1] < 20 or angle[1] > 160:
            return False
        if angle[2] < 0 or angle[2] > 120:
            return False
        return True

    def _is_valid_face_angle(self, angle):
        # print("angle = {}".format(angle))
        if angle[0] * angle[1] * angle[2] == 0:
            return False
        if angle[0] < 60 or angle[0] > 120:
            print("invalid face x angle", angle[0])
            return False
        if angle[1] < 40 or angle[1] > 80:
            print("invalid face y angle", angle[1])
            return False
        if angle[2] < 0 or angle[2] > 60:
            print("invalid face z angle", angle[2])
            return False
        return True

    def _detect_frame(self, color_image, depth_frame):
        src_keypoints, cv_output = self._body_keypoints(color_image)
        frame_center = [self.width/2, self.height/2]
        src_keypoints = K.sort_keypoints(src_keypoints, frame_center)

        person_num = min(src_keypoints.shape[0], self.max_person_one_frame)

        skeletons = []
        for i in range(person_num):
            temp_keypoint = src_keypoints[i, :, :]
            temp_point = self._keypoint_to_point(temp_keypoint, depth_frame)
            face_angle = K.get_face_angle(temp_point, self.mode)
            body_angle = K.get_body_angle(temp_point, self.mode)
            valid_body = self._is_valid_body_angle(body_angle)
            valid_face = self._is_valid_face_angle(face_angle)
            if valid_body and valid_face:
                skeletons.append([temp_keypoint, temp_point])
            else:
                print("invalid angle")
                print("body angle: ", body_angle)
                print("face angle: ", face_angle)
        return skeletons, cv_output

    def _get_max_diou_skeleton(self, target_box, skeletons):
        diou_list = []
        for keypoint, _ in skeletons:
            current_box = K.get_keypoint_box(keypoint)
            current_diou = box_iou(target_box, current_box)
            diou_list.append(current_diou)
        idx = np.argmax(diou_list)
        return skeletons[idx]

    def _reset_target(self):
        self.target['person_id'] = '0'
        self.target['miss_times'] = 0
        self.target['vote_prop'] = []

    def _update_memory(self, color_image, skeletons):
        '''update memory info with current keypoints and points info

        Args:
            color_image: [ndarray], current color frame
            skeletons: [list of list], [keypoint, point]

        Return:
            result_ids: [list of list], [[id1, sim1], [id1, sim1], ...], when
                        the person is not in the memory, the id is set to '0'.

        '''

        self.person_ids = []
        memory_person_ids = self.memory_manager.memory_ids()
        self.memory_manager.reset_memory_state()
        for skeleton in skeletons:
            input_info = self.memory_manager.prepare_input(color_image, skeleton)
            best_person_id, best_sim = self.memory_manager.find_best_match(input_info)
            valid_person = best_person_id in memory_person_ids
            if best_sim > self.sim_thr1 and valid_person:
                self.memory_manager.update_person_memory(best_person_id, input_info)
                if self._update_output_cache(best_person_id):
                    self.person_ids.append([best_person_id, best_sim])
                    memory_person_ids.remove(best_person_id)
                else:
                    self.person_ids.append(['0', 0])
            else:
                if self.memory_manager.update_cache(input_info):
                    self.memory_manager.pop_front_or_not()
                    person_id = self.memory_manager.push_back(input_info)
                    self.person_ids.append([person_id, 1])
                else:
                    self.person_ids.append(['0', 0])
        self.memory_manager.reset_memory_info()

    def _update_output_cache(self, best_person_id, pop_num=4):
        capacity = self.output_cache_params['capacity']
        pop_num = self.output_cache_params['pop_num']
        if len(self.output_cache) > capacity:
            self.output_cache = {}

        if best_person_id not in self.output_cache.keys():
            self.output_cache[best_person_id] = 0
        self.output_cache[best_person_id] += 1

        num = self.output_cache[best_person_id]
        if num > pop_num:
            self.output_cache[best_person_id] -= 1
            return True
        else:
            return False

    def _update_points(self, skeletons, person_ids):
        for (person_id, _), (_, point) in zip(person_ids, skeletons):
            print('person_id: ', person_id)
            if person_id == '0':
                continue
            if person_id not in self.points_seq.keys():
                self.points_seq[person_id] = []
            self.points_seq[person_id].append(point.flatten())

    def _predict(self):
        self.predict_result = {}
        for person_id, point_list in self.points_seq.items():
            if len(point_list) >= self.seq_len:
                point_array = np.array(point_list, dtype='float32')
                point_array = self.dataset._get_location_info(point_array)
                label, score = self._gesture_rec(person_id, point_array)
                self.predict_result[person_id] = [label, score]
                self.points_seq[person_id].pop(0)
                # del self.points_seq[person_id][:5]

    def _update_trigger(self, trigger='hello'):
        trigger_dict = {k: v for k, v in self.predict_result.items()
                        if v[0] == trigger}
        if len(trigger_dict) > 0:
            # predict_results = {person_id: [label, score], ...}
            target = sorted(trigger_dict.items(), key=lambda kv: kv[1][1])
            self.target['person_id'] = target[-1][0]

    def _multi_track(self, color_image, skeletons):
        self._update_memory(color_image, skeletons)
        self._update_points(skeletons, self.person_ids)
        self._predict()
        self._update_trigger(trigger='hello')

    def _single_track(self, color_image, skeletons):
        # print("in single track, target = ", self.target)
        self.person_ids = []
        self.memory_manager.reset_memory_state()
        target_person = self.target['person_id']
        if len(skeletons) > 0:
            memory_info = self.memory_manager.memory[target_person]
            target_box = memory_info['keypoint_box']
            prop_skeleton = self._get_max_diou_skeleton(target_box, skeletons)
            input_info = self.memory_manager.prepare_input(color_image, prop_skeleton)
            sim = self.matcher.match(memory_info, input_info)

            if sim > self.sim_thr1:
                # find track target
                self.memory_manager.update_person_memory(target_person, input_info)
                self.person_ids.append([target_person, sim])
                self.target['miss_times'] = 0
            else:
                self.target['miss_times'] += 1
                max_miss_times = self.target_params['max_miss_times']
                if self.target['miss_times'] >= max_miss_times:
                    # lose track target
                    self._reset_target()
                    self.person_ids.append(['0', 0])
                    return False
                else:
                    return True
            self.memory_manager.reset_memory_info()
            self._update_points([prop_skeleton], self.person_ids)
            self._predict()
            trigger, score = self.predict_result[target_person]
            trigger = self._update_target_vote_prop(trigger)
            if trigger == 'come' and score > self.score_thr:
                print("come: service state.")
                return True
            if trigger == 'wave' and score > self.score_thr:
                self._reset_target()
                print("wave: service canceled.")
                return False
        else:
            self.target['miss_times'] += 1
            max_miss_times = self.target_params['max_miss_times']
            if self.target['miss_times'] >= max_miss_times:
                # lose track target
                self.memory_manager.reset_memory_info()
                self._reset_target()
                return False
            else:
                return True

    def run(self, show=False):
        try:
            self.rs_camera.start_rs_pipe()
        except RuntimeError:
            print("no device found for camera serial: ", self.D435)
            print("available realsense cameras:")
            RsRGBD.show_available_device()
            exit(0)

        frame_number = 0
        while True:
            print('------------------------------')
            frame = self.rs_camera.next_frame()

            if not frame:
                break
            color_image = self.rs_camera.get_color_img(frame)
            depth_frame = frame['depth']
            if frame['number'] == frame_number:
                continue
            frame_number = frame['number']
            time_s = time.time()
            skeletons, cv_output = self._detect_frame(color_image, depth_frame)
            target_person = self.target['person_id']
            if target_person == '0':
                self._multi_track(color_image, skeletons)
            else:
                self._single_track(color_image, skeletons)
            time_e = time.time()
            if show is True:
                cv_output = self._draw_person_ids(cv_output, skeletons)
                cv_output = self._draw_fps(cv_output, time_e-time_s)
                cv_output = self._draw_target(cv_output)
                cv2.imshow("gesture output", cv_output)
                cv2.waitKey(1)
