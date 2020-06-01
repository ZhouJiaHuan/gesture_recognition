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

        pre_path = "gesture_lib/apis/shape_predictor_68_face_landmarks.dat"
        rec_path = "gesture_lib/apis/dlib_face_recognition_resnet_model_v1.dat"

        self.face_pre = dlib.shape_predictor(pre_path)
        self.face_rec = dlib.face_recognition_model_v1(rec_path)
        self.sim_thr1 = 0.35  # for memory
        self.sim_thr2 = 0.65  # for memory cache

    def _extract_feature(self, color_image, keypoint):
        feature = np.zeros([0, 128])
        # Nose, REye, LEye, REar, LEar
        if self.mode == "openpose":
            keypoint = keypoint[[0, 15, 16, 17, 18], :2]
        elif self.mode == "trtpose":
            keypoint = keypoint[[0, 2, 1, 4, 3], :]
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

    def _person_sim(self, memory_info, input_info):
        sim = 0
        feature1 = memory_info['keypoint_feature']
        feature2 = input_info['keypoint_feature']
        if feature1.shape[0] == 0 or feature2.shape[0] == 0:
            return sim
        sim = 1 - euclidean_dis(feature1, feature2)
        return sim
