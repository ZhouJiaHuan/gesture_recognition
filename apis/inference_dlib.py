import sys
sys.path.append(".")

import time
import numpy as np
import cv2
import dlib

from .dlib_face import face_feature, euclidean_dis
from .inference import Inference


class InferenceDlib(Inference):
    '''inference of body keypoints model for video from realsense
    '''

    def __init__(self, *args, **kwargs):
        super(InferenceDlib, self).__init__(*args, **kwargs)

        pre_path = "./apis/shape_predictor_5_face_landmarks.dat"
        rec_path = "./apis/dlib_face_recognition_resnet_model_v1.dat"

        self.face_pre = dlib.shape_predictor(pre_path)
        self.face_rec = dlib.face_recognition_model_v1(rec_path)

    def _get_face_feature(self, color_image, src_keypoint):
        feature = np.zeros([0, 128])
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

        if img.shape[0] * img.shape[1] > 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            det = dlib.rectangle(0, 0, w, h)

            feature, chip = face_feature(self.face_pre, self.face_rec, img, det)
            cv2.imshow("aligned image", chip)
            cv2.waitKey(1)
        return feature

    def _feature_similarity(self, feature1, feature2):
        sim = 0
        if feature1.shape[0] == 0 or feature2.shape[0] == 0:
            return sim
        sim = 1 - euclidean_dis(feature1, feature2)
        return sim

    def _person_sim(self, person_1, person_2):
        return self._feature_similarity(person_1[0], person_2[0])

    def _is_in_memory(self, sim, sim_thr=0.35):
        return sim > sim_thr

    def _update_memory(self, color_image, keypoints_list, points_list):
        result_ids = []
        # print("keypoints num: {}".format(len(keypoints_list)))
        for keypoint, point in zip(keypoints_list, points_list):
            face_feature = self._get_face_feature(color_image, keypoint)
            input_info = [face_feature]
            best_person_id, best_sim = self._find_best_match(input_info)

            if best_person_id in result_ids:
                continue

            if self._is_in_memory(best_sim):
                if self._update_output_cache(best_person_id):
                    result_ids.append([best_person_id, best_sim])
                else:
                    result_ids.append(['0', 0])
            else:
                if self._update_memory_cache(input_info, sim_thr=0.65):
                    self.memory['max_id'] += 1
                    self.memory_count += 1
                    person_id = 'person_' + str(self.memory['max_id'])
                    self.memory[person_id] = input_info
                    result_ids.append([person_id, 1])
                else:
                    result_ids.append(['0', 0])

        return result_ids
