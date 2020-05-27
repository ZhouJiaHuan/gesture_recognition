import sys
sys.path.append(".")

import time
import os
import numpy as np
import cv2
from collections import Counter
import torch
from torch.nn.functional import softmax

from mmcv import Config
from dataset import build_dataset, OpenposeExtractor, TrtposeExtractor
from model import build_model
from .point_seq import is_static, get_body_angle, get_face_angle
from .point_seq import post_process_wave, post_process_come, post_process_hello


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
                 pose_json_path=None,
                 feature='surf'):
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

        # configure feature extractor for tracking and re-ID
        if feature == 'surf':
            self.face_feature = cv2.xfeatures2d.SURF_create(200, extended=False, upright=0)
            self.body_feature = cv2.xfeatures2d.SURF_create(100, extended=False, upright=0)
        elif feature == 'akaze':
            self.face_feature = cv2.AKAZE_create()
            self.body_feature = cv2.AKAZE_create()
        else:
            raise
        index_param = dict(algorithm=0, trees=5)
        search_param = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_param, search_param)
        self.max_person_one_frame = 5

        # init params for smoothing result
        self.predict = {}
        self.smooth_count = 0

        # init memory and cache
        self.proposal_person = {}
        self.memory = {'max_id': 0}
        self.memory_count = 0
        self.memory_cache = []
        self.memory_cache_count = 0

        self.output_cache = {}
        self.output_cache_size = 20
        self.max_num = 0
        self.points_seq = {}

    def _body_keypoints(self, color_image):
        '''detect body keypoints from a frame

        Args:
            frame: one frame from realsense, including color and depth frame info

        Return:
            pose keypoints array, shape (N, keypoint_num, :), (x, y) or (x, y, score)

        '''
        color_image = cv2.resize(color_image, (self.pose_w, self.pose_h))
        keypoints, cv_output = self.extractor.body_keypoints(color_image)
        if keypoints.shape[0] > 0:
            keypoints[:, :, 0] = keypoints[:, :, 0] * self.pose_wscale
            keypoints[:, :, 1] = keypoints[:, :, 1] * self.pose_hscale
        cv_output = cv2.resize(cv_output, (self.width, self.height))

        return keypoints, cv_output

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

    def _draw_person_ids(self, cv_output, keypoints, person_ids, predict_result):
        '''
        Args:
            keypoints: shape (N, 25, 3) [x, y, score]
            person_ids: list, len = N
        '''
        result_img = cv_output.copy()
        for idx, (person_id, sim) in enumerate(person_ids):
            sim = round(sim, 3)
            keypoint = keypoints[idx, :, :]
            keypoint = keypoint[keypoint[:, -1] > 1e-2]
            x1 = int(np.min(keypoint[:, 0]))
            y1 = int(np.min(keypoint[:, 1]))
            x2 = int(np.max(keypoint[:, 0]))
            y2 = int(np.max(keypoint[:, 1]))
            cv2.rectangle(result_img, (x1,y1), (x2,y2), color=(255,0,0), thickness=2)
            cv2.putText(result_img, person_id + ' '+str(sim), (x1,y1+20), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
            if person_id in predict_result.keys():
                predict_label = predict_result[person_id][0]
                score = np.round(predict_result[person_id][1], 2)
                if predict_label == "others":
                    continue
                ss = predict_label + ' ' + str(score)
                cv2.putText(result_img, ss, (x1,y1+40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
        return result_img

    def _draw_fps(self, image, t):
        image = np.array(image, dtype=np.uint8)
        speed = str(int(1 / t)) + ' FPS'
        cv2.putText(image, speed, (image.shape[1]-120, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
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
        if angle[1] < 50 or angle[1] > 80:
            print("invalid face y angle", angle[1])
            return False
        if angle[2] < 0 or angle[2] > 60:
            print("invalid face z angle", angle[2])
            return False
        return True

    def _get_body_feature(self, color_image, src_keypoint):
        feature = np.zeros([0, self.body_feature.descriptorSize()])
        if self.mode == "openpose":
            keypoint = src_keypoint[[2, 5, 8], :]
            keypoint = keypoint[keypoint[:, -1] > 0, :2]
            if keypoint.shape[0] < 3:
                return feature
        else:
            keypoint = src_keypoint[[6, 5, 12, 11], :]
            keypoint = keypoint[keypoint[:, -1] > 0, ]
            if keypoint.shape[0] < 4:
                return feature
            keypoint[2, ] = (keypoint[2, ] + keypoint[3, ]) / 2
            keypoint = keypoint[:3, :]

        keypoint = np.float32(keypoint)
        rect = cv2.minAreaRect(keypoint)
        box = cv2.boxPoints(rect)
        x1 = int(np.floor(np.min(box[:, 0])))
        y1 = int(np.floor(np.min(box[:, 1])))
        x2 = int(np.ceil(np.max(box[:, 0])))
        y2 = int(np.ceil(np.max(box[:, 1])))
        img = color_image[y1:y2, x1:x2]
        if img.shape[0] * img.shape[1] > 0:
            cols = x2 - x1
            rows = y2 - y1
            src = np.float32(keypoint - [x1, y1])
            dst = np.float32([[0, 0], [cols, 0], [cols/2, rows]])
            M = cv2.getAffineTransform(src, dst)
            img_affine = cv2.warpAffine(img, M, (cols, rows))
            t_s = time.time()
            img_affine = cv2.resize(img_affine, (80, 120))
            _, temp_feature = self._feature_extractor(self.body_feature, img_affine)
            feature = temp_feature if temp_feature is not None else feature
            # cv2.imshow("src body area", img[:,:,::-1])
            # cv2.imshow("dst body area", img_affine[:,:,::-1])
            # cv2.waitKey(1)
            feature_time = time.time() - t_s
            # print('body feature time = {:.3f}'.format(feature_time*1000))  # about 0.6 ms
        return feature

    def _get_face_feature(self, color_image, src_keypoint):
        feature = np.zeros([0, self.face_feature.descriptorSize()])
        # Nose, REye, LEye, REar, LEar
        if self.mode == "openpose":
            keypoint = src_keypoint[[0, 15, 16, 17, 18], :2]
        elif self.mode == "trtpose":
            keypoint = src_keypoint[[0, 2, 1, 4, 3], :]
        else:
            raise

        if keypoint[0, 0] * keypoint[1, 0] * keypoint[2, 0] == 0:
            return feature

        if keypoint[0, 0] < keypoint[1, 0] or keypoint[0, 0] > keypoint[2, 0]:
            return feature

        height, width = color_image.shape[:2]
        x1 = keypoint[3, 0] if keypoint[3, 0] > 0 else keypoint[1, 0]*0.98
        x2 = keypoint[4, 0] if keypoint[4, 0] > 0 else keypoint[2, 0]*1.02
        x1, x2 = max(0, int(x1)), min(width, int(x2))
        w, h = x2 - x1, int((x2 - x1) * 1.1)
        y1 = keypoint[0, 1] - h / 2
        y2 = keypoint[0, 1] + h / 2
        y1, y2 = max(0, int(y1)), min(height, int(y2))
        img = color_image[y1:y2, x1:x2]
        print("img shape = ", img.shape)
        cv2.rectangle(color_image, (x1, y1), (x2, y2), (0, 0, 255))

        if img.shape[0] * img.shape[1] > 0:
            src = np.float32(keypoint[:3, ] - [x1, y1])
            t_s = time.time()
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            angle = np.arctan2(src[2, 1]-src[1, 1], src[2, 0]-src[1, 0]) * 180 / np.pi
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img_affine = cv2.warpAffine(img, M, (w, h))
            img_affine = cv2.resize(img_affine, (96, 112))
            t_s = time.time()
            _, temp_feature = self._feature_extractor(self.face_feature, img_affine)
            feature_time = time.time() - t_s
            feature = temp_feature if temp_feature is not None else feature
            cv2.imshow("src face area", img[:,:,::-1])
            cv2.imshow("dst face area", img_affine[:,:,::-1])
            cv2.waitKey(1)
            
            print('face feature time = {:.3f}'.format(feature_time*1000))  # about 0.6 ms
        return feature

    def _feature_extractor(self, extractor, img):
        kp, des = extractor.detectAndCompute(img, None)
        return kp, des

    def _single_face_detector(self, img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # [x, y, w, h]
        faces = self.face_model.detectMultiScale(img_gray,
                                                 scaleFactor=1.1,
                                                 minNeighbors=1,
                                                 minSize=(5, 5))
        if len(faces) == 0:
            return np.zeros([0, 4])
        else:
            return np.array(faces[0])

    def _feature_similarity(self, feature1, feature2):
        similarity = 0
        match_ids = []
        if feature1.shape[0] < 2 or feature2.shape[0] < 2:
            return similarity, match_ids
        feature1 = np.asarray(feature1, dtype=np.float32)
        feature2 = np.asarray(feature2, dtype=np.float32)
        matches = self.matcher.knnMatch(feature2, feature1, k=2)

        num = 0
        match_ids = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                num += 1
                match_ids.append([m.trainIdx, m.queryIdx])
        similarity = num / len(matches)
        # print("match_ids = ", match_ids)
        return similarity, match_ids

    def _person_sim(self, person_1, person_2):
        face_feature1, body_feature1 = person_1
        face_feature2, body_feature2 = person_2
        # print(face_feature1.shape, face_feature2.shape)
        # print(body_feature1.shape, body_feature2.shape)
        face_sim, face_matches = self._feature_similarity(face_feature1, face_feature2)
        body_sim, body_matches = self._feature_similarity(body_feature1, body_feature2)
        sim = 1 * face_sim + 0.0 * body_sim
        matches = [face_matches, body_matches]
        # print("face sim: {:.3f}, body sim: {:.3f}, total sim: {:.3f}".format(face_sim, body_sim, sim))

        return sim, matches

    def _find_best_match(self, input_info):
        best_person_id = '0'
        best_sim = 0
        best_matches = []
        if self.memory['max_id'] == 0:
            return best_person_id, best_sim, best_matches
        # print("memory['max_id'] = ", self.memory['max_id'])
        for i in range(self.memory['max_id']):
            person_id = 'person_'+str(i+1)
            memory_info = self.memory[person_id]
            temp_sim, temp_matches = self._person_sim(memory_info, input_info)

            if temp_sim > best_sim:
                best_sim = temp_sim
                best_person_id = person_id
                best_matches = temp_matches
        return best_person_id, best_sim, best_matches

    def _is_in_memory(self, sim, sim_thr=0.2):
        return sim > sim_thr

    def _update_person_memory(self, input_info, person_id, sim, matches):
        return False  # test
        if person_id not in self.memory.keys():
            return False
        if sim <= 0 or sim > 1:
            return False
        face_matches, body_matches = matches
        if len(face_matches) > 0:
            face_matches = np.int32(face_matches)
            face_update = input_info[0][face_matches[:, 1]] * sim
            face_update += self.memory[person_id][0][face_matches[:, 0]] * (1-sim)
            self.memory[person_id][0][face_matches[:, 0]] = face_update
        if len(body_matches) > 0:
            body_matches = np.int32(body_matches)
            body_update = input_info[1][body_matches[:, 1]] * sim
            body_update += self.memory[person_id][1][body_matches[:, 0]] * (1-sim)
            self.memory[person_id][1][body_matches[:, 0]] = body_update
        return True

    def _update_memory(self, color_image, keypoints_list, points_list):
        result_ids = []
        # print("keypoints num: {}".format(len(keypoints_list)))
        for keypoint, point in zip(keypoints_list, points_list):
            face_feature = self._get_face_feature(color_image, keypoint)
            body_feature = self._get_body_feature(color_image, keypoint)
            input_info = [face_feature, body_feature]
            best_person_id, best_sim, best_matches = self._find_best_match(input_info)

            if best_person_id in result_ids:
                continue

            if self._is_in_memory(best_sim):
                self._update_person_memory(input_info, best_person_id, best_sim, best_matches)
                if self._update_output_cache(best_person_id):
                    result_ids.append([best_person_id, best_sim])
                else:
                    result_ids.append(['0', 0])
            else:
                if self._update_memory_cache(input_info):
                    self.memory['max_id'] += 1
                    self.memory_count += 1
                    person_id = 'person_' + str(self.memory['max_id'])
                    self.memory[person_id] = input_info
                    result_ids.append([person_id, 1])
                else:
                    result_ids.append(['0', 0])

        return result_ids

    def _update_memory_cache(self, input_info, clean_num=20, pop_num=4):
        self.memory_cache_count += 1
        if self.memory_cache_count > clean_num:
            self.memory_cache_count = 0
            self.memory_cache = []

        for idx, temp_info in enumerate(self.memory_cache):
            temp_sim, _ = self._person_sim(temp_info[:-1], input_info)
            if temp_sim > 0.9:
                self.memory_cache[idx][-1] += 1
                self.memory_cache[idx][:len(input_info)] = input_info
                count = self.memory_cache[idx][-1]
                if count >= pop_num:
                    self.memory_cache.pop(idx)
                    return True

        face_feature, body_feature = input_info
        if face_feature.shape[0] > 0 and body_feature.shape[0] > 0:
            self.memory_cache.append([face_feature, body_feature, 1])
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

    def _update_points(self, current_points, person_ids):
        for (person_id, _), point in zip(person_ids, current_points):
            if person_id == '0':
                continue
            if person_id not in self.points_seq.keys():
                self.points_seq[person_id] = []
            self.points_seq[person_id].append(point.flatten())

    def _predict(self):
        predict_result = {}
        for person_id, point_list in self.points_seq.items():
            if len(point_list) >= self.seq_len:
                point_array = np.array(point_list, dtype='float32')
                point_array = self.dataset._get_location_info(point_array)
                label, score = self._predict_one_point(person_id, point_array)
                predict_result[person_id] = [label, score]
                self.points_seq[person_id].pop(0)
                # del self.points_seq[person_id][:5]
        return predict_result

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

            depth_frame, color_frame = self.extractor._get_aligned_frame(frames)

            if (not depth_frame) or (not color_frame):
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # get 3-D points info with openpose
            time_s = time.time()
            keypoints, cv_output = self._body_keypoints(color_image)
            keypoints = self._sort_keypoints(keypoints)

            person_num = min(keypoints.shape[0], self.max_person_one_frame)

            current_points = []
            for i in range(person_num):
                temp_point = self.extractor.keypoint_to_point(keypoints[i, :, :], depth_frame)
                face_angle = get_face_angle(temp_point, self.mode)
                body_angle = get_body_angle(temp_point, self.mode)
                valid_body = self._is_valid_body_angle(body_angle)
                valid_face = self._is_valid_face_angle(face_angle)
                if valid_body and valid_face:
                    current_points.append(temp_point)
                else:
                    print("invalid angle")
                    print("body angle: ", body_angle)
                    print("face angle: ", face_angle)
            person_ids = self._update_memory(color_image, keypoints, current_points)
            if len(person_ids) > 0:
                self._update_points(current_points, person_ids)
            # print(person_ids)
            t_s = time.time()
            predict_result = self._predict()
            # print("gesture time: {:.3f}".format((time.time()-t_s)*1000)) # 3 ms / person
            # print("predict_result = ", predict_result)
            cv_output = self._draw_person_ids(cv_output, keypoints, person_ids, predict_result)

            time_e = time.time()
            # result visualization
            if show is True:
                cv_output = self._draw_fps(cv_output, time_e-time_s)
                cv2.imshow("gesture output", cv_output[:, :, ::-1])
                cv2.waitKey(1)
        if self.op_wrapper is not None:
            self.op_wrapper.stop()
