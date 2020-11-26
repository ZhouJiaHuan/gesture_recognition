import time
import numpy as np
import cv2
import os
import torch
from collections import Counter
import matplotlib.pyplot as plt
import gesture_lib.ops.keypoint as K
from torch.nn.functional import softmax
from mpl_toolkits.mplot3d import Axes3D
from gesture_lib.datasets import build_dataset
from gesture_lib.datasets import Body3DExtractor
from gesture_lib.models import build_model, build_matcher
from gesture_lib.models.generator import SkeletonGenerator, visSkeleton3D
from gesture_lib.ops.realsense import RsRGBD, RsFishEye
from gesture_lib.ops.yaml_utils import parse_yaml
from gesture_lib.models import MemoryManager


class InferenceMulticam(object):

    def __init__(self, cfg_path, checkpoints):
        assert os.path.splitext(checkpoints)[-1] == '.pth'
        cfg = parse_yaml(cfg_path)
        core_cfg = parse_yaml("gesture_lib/configs.yaml")

        # configure dataset
        data_cfg = cfg.dataset
        self.cls_names = data_cfg.test.cls_names
        self.body_names = data_cfg.test.body_names
        self.dataset = build_dataset(data_cfg.test)
        self.idx_list = self.dataset.idx_list
        self.transforms = self.dataset.transforms

        # realsense D435 and T265
        self.width = core_cfg.width
        self.height = core_cfg.height
        self.size = max(self.width, self.height)
        self.D435 = core_cfg.camera_id.D435
        self.rgbd = RsRGBD(self.width, self.height, serial_number=self.D435)
        self.T265 = core_cfg.camera_id.T265
        self.fisheye = RsFishEye(serial_number=self.T265)

        # pose estimation
        pose_params = core_cfg.pose_params
        self.mode = pose_params.mode
        self.pose_w = pose_params.pose_w
        self.pose_h = pose_params.pose_h
        self.extractor = Body3DExtractor()
        self.generator = SkeletonGenerator(self.extractor)

        # perspective matrix
        self.M = np.loadtxt("gesture_lib/data/perspMatrix.txt")

        # cache for output (for smoothing the noise)
        self.output_cache = {}
        self.output_cache_params = core_cfg.output_cache_params

        # person info
        self.person_ids = []  # for tracking: [id, sim]

        # configure matcher and memory for tracking
        matcher_params = core_cfg.matcher_params
        matcher_params.mode = core_cfg.pose_params.mode
        self.matcher = build_matcher(matcher_params)
        memory_params = core_cfg.memory_params
        memory_params.matcher = self.matcher
        self.memory_manager = MemoryManager(**memory_params)
        self.sim_thr1 = memory_params.sim_thr1
        self.sim_thr2 = memory_params.sim_thr2

        # configure gesture recognition
        self.max_person_one_frame = core_cfg.max_person_one_frame
        self.seq_len = core_cfg.gesture_params.seq_len
        self.score_thr = core_cfg.gesture_params.score_thr
        self._configure_gesture_rec(cfg.model, checkpoints)

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

    def start_rs_pipe(self):
        try:
            self.rgbd.start_rs_pipe()
        except RuntimeError:
            print("no device found for D435 camera serial: ", self.D435)
            exit(0)
        try:
            self.fisheye.start_rs_pipe()
        except RuntimeError:
            print("no device found for T265 camera serial: ", self.T265)
            exit(0)

    def close_rs_pipe(self):
        try:
            self.rgbd.close_rs_pipe()
            self.fisheye.close_rs_pipe()
        except Exception as e:
            print(e)

    def _body_keypoints(self, fish_img):
        '''detect body keypoints from a frame

        Args:
            frame: one frame from realsense, including color
                and depth frame info

        Return:
            keypoints: [ndarray], pose keypoints array,
                shape (N, keypoint_num, :), (x, y) or (x, y, score)
            cv_output: [ndarray]: output image with keypoints info.

        '''
        kp, out = self.extractor.body_keypoints(fish_img)
        return kp, out

    def _draw_fps(self, image, t):
        image = np.array(image, dtype=np.uint8)
        speed = str(int(1 / t)) + ' FPS'
        cv2.putText(image, speed, (image.shape[1]-120, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return image

    def _draw_keypoint(self, image, skeletons, rgbd=True):
        h, w = image.shape[:2]
        out = image.copy()
        num = len(skeletons)
        for i in range(num):
            keypoint = skeletons[i][1] if rgbd else skeletons[i][0]
            for j in range(keypoint.shape[0]):
                point = (int(keypoint[j, 0]), int(keypoint[j, 1]))
                cv2.circle(out, point, 5, (0, 255, 0), -1)
        return out

    def _draw_point(self, ax, skeletons):
        points = np.array([skeleton[-1] for skeleton in skeletons])
        visSkeleton3D(ax, points)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([0, 3])

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

    def _post_process(self, points_array, label):
        if label == "wave":
            return K.post_process_wave(points_array, self.mode)
        elif label == "come":
            return K.post_process_come(points_array, self.mode)
        elif label == "hello":
            return K.post_process_hello(points_array, self.mode)
        else:
            return True

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

    def _detect_frame(self, fish_img, rgbd_frame):
        fish_kps, cv_output = self._body_keypoints(fish_img)
        frame_center = [self.width/2, self.height/2]
        fish_kps = K.sort_keypoints(fish_kps, frame_center)
        rgbd_img = self.rgbd.get_color_img(rgbd_frame)
        fish_img = cv2.cvtColor(fish_img, cv2.COLOR_BGR2GRAY)
        rgbd_img = cv2.cvtColor(rgbd_img, cv2.COLOR_BGR2GRAY)
        person_num = min(fish_kps.shape[0], self.max_person_one_frame)

        skeletons = []
        for i in range(person_num):
            fish_kp = fish_kps[i, :, :]
            rgbd_kp = self.generator.trans_kp_fish2rgbd(fish_kp, fish_img, rgbd_img)
            src_point = self.generator.keypoint_to_point(rgbd_frame, rgbd_kp)
            dst_point = self.generator.generate(rgbd_frame, src_point, fish_kp, rgbd_kp)
            # face_angle = K.get_face_angle(dst_point, self.mode)
            # body_angle = K.get_body_angle(dst_point, self.mode)
            # valid_body = self._is_valid_body_angle(body_angle)
            # valid_face = self._is_valid_face_angle(face_angle)
            skeletons.append([fish_kp, rgbd_kp, dst_point])
            # if valid_body and valid_face:
            #     skeletons.append([fish_kp, rgbd_kp, dst_point])
            # else:
            #     print("invalid angle")
            #     print("body angle: ", body_angle)
            #     print("face angle: ", face_angle)
        return skeletons, cv_output

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

    def _multi_track(self, color_image, skeletons):
        sks = [[sk[0], sk[-1]] for sk in skeletons]
        self._update_memory(color_image, sks)
        self._update_points(sks, self.person_ids)
        self._predict()

    def run(self, show=False, fisheye="left"):
        self.start_rs_pipe()
        self.fisheye.init_undistort_rectify(self.size)
        if show:
            fig = plt.figure()
            ax = Axes3D(fig)
            # ax = fig.add_subplot(111, projection='3d')

        try:
            while True:
                rgbd_frame = self.rgbd.next_frame()
                fish_frame = self.fisheye.next_frame()
                if (rgbd_frame is None) or (fish_frame is None):
                    print("camera disconnected!")
                    print("avaliable realsense camera: ")
                    self.fisheye.show_available_device()
                    break
                time_s = time.time()
                rgbd_img = self.rgbd.get_color_img(rgbd_frame)
                fish_img = self.fisheye.get_color_img(fish_frame, fisheye)
                # fish_img = fish_img[80:560, :, :]  # keep same size with rgbd camera

                skeletons, output1 = self._detect_frame(fish_img, rgbd_frame)
                self._multi_track(fish_img, skeletons)
                
                time_e = time.time()

                if show:
                    output1 = self._draw_fps(output1, time_e-time_s)
                    output1 = self._draw_person_ids(output1, skeletons)
                    output2 = self._draw_keypoint(rgbd_img, skeletons)
                    # plt.cla()
                    # self._draw_point(ax, skeletons)
                    cv2.imshow("fish color", output1)
                    cv2.imshow("rgbd color", output2)
                    # plt.pause(0.000001)

                    if cv2.waitKey(1) == ord('q'):
                        break
        except RuntimeError:
            print("camera disconnected!")
        finally:
            self.close_rs_pipe()
