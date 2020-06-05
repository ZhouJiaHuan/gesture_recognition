# import sys
# sys.path.append(".")

import time
import os
import numpy as np
import cv2
from abc import abstractmethod
import torch
from torch.nn.functional import softmax

from mmcv import Config
from ..dataset import build_dataset, OpenposeExtractor, TrtposeExtractor
from ..model import build_model
from .point_seq import is_static, get_body_angle, get_face_angle
from .point_seq import post_process_wave, post_process_come, post_process_hello
from .point_seq import get_keypoint_box, get_location


class Inference(object):
    '''inference of body keypoints model for video from realsense
    '''

    def __init__(self,
                 cfg_path,
                 checkpoints,
                 seq_len,
                 op_model=None,
                 model_pose='BODY_25',
                 trt_model=None,
                 pose_json_path=None):
        assert os.path.splitext(checkpoints)[-1] == '.pth'
        cfg = Config.fromfile(cfg_path)
        print(dict(cfg))

        # configure dataset
        data_cfg = cfg.dataset
        self.cls_names = data_cfg.test.cls_names
        self.body_names = data_cfg.test.body_names
        self.dataset = build_dataset(data_cfg.test)
        self.idx_list = self.dataset.idx_list
        self.transforms = self.dataset.transforms

        # configure gesture model
        self.seq_len = seq_len
        self.gesture_model = build_model(cfg.model).cuda()
        self.gesture_model.load_state_dict(torch.load(checkpoints))
        self.gesture_model.eval()

        # configure pose model
        self.width = 640
        self.height = 480
        self.pose_w = 224
        self.pose_h = 224
        self.op_wrapper = None
        if op_model is not None:
            self.mode = "openpose"
            self.extractor = OpenposeExtractor(op_model, model_pose)
            self.pose_wscale = self.width / self.pose_w
            self.pose_hscale = self.height / self.pose_h
            self.dim = 4  # (x, y, z, score)
            self.op_wrapper = self.extractor.op_wrapper
            self.datum = self.extractor.datum
        elif trt_model is not None and pose_json_path is not None:
            self.mode = "trtpose"
            self.extractor = TrtposeExtractor(trt_model,
                                              pose_json_path)
            self.pose_wscale = 1
            self.pose_hscale = 1
            self.dim = 3  # (x, y, z)
        else:
            raise

        # configure realsense camera
        self.rs_pipe = self.extractor.rs_pipe
        self.rs_cfg = self.extractor.rs_cfg

        # init params for smoothing result
        self.predict = {}
        self.smooth_count = 0

        # init memory and cache
        self.sim_thr1 = 0.2  # for memory
        self.sim_thr2 = 0.8  # for memory cache
        self.max_person_one_frame = 5  # maximum person in a frame
        self.memory = {}  # memory info
        self.memory_max_id = 100  # maximum person id for record in memory
        self.memory_capacity = 5  # memory capacity
        self.memory_count = 0  # total person num memory recorded until now
        self.memory_pop_id = 1  # person id for pop when capacity reached
        self.memory_cache = []  # cache before saving to memory
        self.memory_cache_count = 0  # visit times of cache (for clean)

        self.output_cache = {}  # cache for output (for smoothing the noise)
        self.output_cache_size = 20  # cache capacity (for clean)

        # result of current frame
        self.person_ids = []  # for tracking: [id, sim]
        self.points_seq = {}  # input sequence for gesture model
        self.predict_result = {}  # gesture model output

    def _get_aligned_frame(self, frames):
        return self.extractor._get_aligned_frame(frames)

    def _body_keypoints(self, color_image):
        '''detect body keypoints from a frame

        Args:
            frame: one frame from realsense, including color
                and depth frame info

        Return:
            keypoints: [ndarray], pose keypoints array,
                shape (N, keypoint_num, :), (x, y) or (x, y, score)
            cv_output: [ndarray]: output image with keypoints info.

        '''
        color_image = cv2.resize(color_image, (self.pose_w, self.pose_h))
        keypoints, cv_output = self.extractor.body_keypoints(color_image)
        if keypoints.shape[0] > 0:
            keypoints[:, :, 0] = keypoints[:, :, 0] * self.pose_wscale
            keypoints[:, :, 1] = keypoints[:, :, 1] * self.pose_hscale
        cv_output = cv2.resize(cv_output, (self.width, self.height))

        return keypoints, cv_output

    def _keypoint_to_point(self, keypoint, depth_frame):
        point = self.extractor.keypoint_to_point(keypoint, depth_frame)
        return point

    def _sort_keypoints(self, keypoints):
        '''sort the keypoints based on the distance to frame center
        '''
        if keypoints.shape[0] == 0:
            return keypoints
        keypoints_center = np.mean(keypoints[:, :, :2], axis=1)
        keypoints_center = keypoints_center - [self.width, self.height]
        d_to_center = keypoints_center[:, 0] ** 2 + keypoints_center[:, 1] ** 2
        order = np.argsort(d_to_center)
        return keypoints[order]

    def _post_process(self, points_array, label):
        if label == "wave":
            return post_process_wave(points_array, self.mode)
        elif label == "come":
            return post_process_come(points_array, self.mode)
        elif label == "hello":
            return post_process_hello(points_array, self.mode)
        else:
            return True

    def _predict_one_point(self, person_id, point_array):
        predict_label = "others"
        score = 0.0
        if is_static(point_array):
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
        if score > 0.7:
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

    def _draw_person_ids(self, cv_output, skeletons):
        '''
        Args:
            keypoints: shape (N, 25, 3) [x, y, score]
            person_ids: list, len = N
        '''
        result_img = cv_output.copy()
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
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return image

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
        src_keypoints = self._sort_keypoints(src_keypoints)

        person_num = min(src_keypoints.shape[0], self.max_person_one_frame)

        skeletons = []
        for i in range(person_num):
            temp_keypoint = src_keypoints[i, :, :]
            temp_point = self._keypoint_to_point(temp_keypoint, depth_frame)
            face_angle = get_face_angle(temp_point, self.mode)
            body_angle = get_body_angle(temp_point, self.mode)
            # valid_body = self._is_valid_body_angle(body_angle)
            valid_face = self._is_valid_face_angle(face_angle)
            if valid_face:
                skeletons.append([temp_keypoint, temp_point])
            else:
                print("invalid angle")
                print("body angle: ", body_angle)
                print("face angle: ", face_angle)
        return skeletons, cv_output

    @abstractmethod
    def _extract_feature(self, color_image, keypoint):
        '''extract feature from color image with specified keypoint info.
        '''
        raise NotImplementedError

    @abstractmethod
    def _person_sim(self, memory_info, input_info):
        ''' compute the feature similarity based on memory and input_info

        the memory_info and input_info are dicts, which at least contain
        following infomation:
        - keypoint: keypoint array, for trtpose, shape 18x2
        - point: point array, for trtpose, shape 18x2
        - keypoint_box: minimum bounding rectangle, (x1, y1, x2, y2)
        - point_center: 3-D location (x, y, z)
        - keypoint_feature: feature vector extracted from color image with
            keypoint info. The type is decided by `_extract_feature`
        '''
        raise NotImplementedError

    def _find_best_match(self, input_info):
        best_person_id = '0'
        best_sim = 0

        if self.memory_count == 0:
            return best_person_id, best_sim

        for person_id, memory_info in self.memory.items():
            temp_sim = self._person_sim(memory_info, input_info)
            if temp_sim > best_sim:
                best_sim = temp_sim
                best_person_id = person_id
        return best_person_id, best_sim

    def _reset_memory_state(self):
        for person_id in self.memory.keys():
            self.memory[person_id]['visible'] = False

    def _reset_memory_box(self):
        for person_id in self.memory.keys():
            visible = self.memory[person_id]['visible']
            if visible is False:
                self.memory[person_id]['keypoint_box'] = np.zeros(4)

    def _prepare_input_info(self, color_image, keypoint, point):
        input_info = {}
        keypoint_feature = self._extract_feature(color_image, keypoint)
        point_center = get_location(point)
        input_info['keypoint'] = keypoint
        input_info['point'] = point
        input_info['keypoint_box'] = get_keypoint_box(keypoint)
        input_info['point_center'] = point_center
        input_info['keypoint_feature'] = keypoint_feature
        input_info['visible'] = True
        return input_info

    def _update_person_memory(self, person_id, input_info):
        assert person_id in self.memory.keys()
        self.memory[person_id]['keypoint'] = input_info['keypoint']
        self.memory[person_id]['point'] = input_info['point']
        self.memory[person_id]['keypoint_box'] = input_info['keypoint_box']
        self.memory[person_id]['point_center'] = input_info['point_center']
        self.memory[person_id]['visible'] = input_info['visible']

    def _pop_memory_or_not(self):
        if len(self.memory) >= self.memory_capacity:
            self.memory.pop(str(self.memory_pop_id), 0)
            self.memory_pop_id = self.memory_pop_id % self.memory_max_id + 1

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
        memory_person_ids = list(self.memory.keys())
        self._reset_memory_state()
        for keypoint, point in skeletons:
            input_info = self._prepare_input_info(color_image, keypoint, point)
            best_person_id, best_sim = self._find_best_match(input_info)

            if best_sim > self.sim_thr1:
                self._update_person_memory(best_person_id, input_info)
                if self._update_output_cache(best_person_id):
                    self.person_ids.append([best_person_id, best_sim])
                    memory_person_ids.remove(best_person_id)
                else:
                    self.person_ids.append(['0', 0])
            else:
                if self._update_memory_cache(input_info):
                    self._pop_memory_or_not()
                    person_id = str(self.memory_count % self.memory_max_id + 1)
                    self.memory[person_id] = input_info
                    self.person_ids.append([person_id, 1])
                    self.memory_count += 1
                else:
                    self.person_ids.append(['0', 0])
        self._reset_memory_box()

    def _update_memory_cache(self, input_info, clean_num=20, pop_num=4):
        self.memory_cache_count += 1
        if self.memory_cache_count > clean_num:
            self.memory_cache_count = 0
            self.memory_cache = []

        for idx, temp_info in enumerate(self.memory_cache):
            temp_sim = self._person_sim(temp_info[0], input_info)
            if temp_sim > self.sim_thr2:
                self.memory_cache[idx][1] += 1
                self.memory_cache[idx][0] = input_info
                count = self.memory_cache[idx][1]
                if count >= pop_num:
                    self.memory_cache.pop(idx)
                    return True

        self.memory_cache.append([input_info, 1])
        return False

    def _update_output_cache(self, best_person_id, clean_num=20, pop_num=4):
        if len(self.output_cache) > self.output_cache_size:
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
                label, score = self._predict_one_point(person_id, point_array)
                self.predict_result[person_id] = [label, score]
                self.points_seq[person_id].pop(0)
                # del self.points_seq[person_id][:5]

    def infer_pipeline(self, show=False):
        '''inference pipeline for frames stream from realsense
        '''
        self.rs_pipe.start(self.rs_cfg)
        if self.op_wrapper is not None:
            self.op_wrapper.start()

        frame_number = 0
        while True:
            print('------------------------------')
            # get a frame of color image and depth image
            frames = self.rs_pipe.wait_for_frames(timeout_ms=5000)
            current_frame_num = frames.get_frame_number()
            if current_frame_num == frame_number:
                continue
            else:
                frame_number = current_frame_num
            depth_frame, color_frame = self._get_aligned_frame(frames)
            if (not depth_frame) or (not color_frame):
                continue

            # get 3-D points info with openpose
            color_image = np.asanyarray(color_frame.get_data())
            time_s = time.time()
            skeletons, cv_output = self._detect_frame(color_image, depth_frame)
            self._update_memory(color_image, skeletons)
            self._update_points(skeletons, self.person_ids)
            self._predict()
            time_e = time.time()

            # result visualization
            if show is True:
                cv_output = self._draw_person_ids(cv_output, skeletons)
                cv_output = self._draw_fps(cv_output, time_e-time_s)
                cv2.imshow("gesture output", cv_output[:, :, ::-1])
                cv2.waitKey(1)
        if self.op_wrapper is not None:
            self.op_wrapper.stop()
