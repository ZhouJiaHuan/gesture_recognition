import numpy as np
import cv2
import dlib
from gesture_lib.ops import face_feature, euclidean_dis
from .base_matcher import BaseMatcher
from gesture_lib.registry import MATCHERS


@MATCHERS.register_module(name="DlibMatcher")
class DlibMatcher(BaseMatcher):
    '''Matcher with Dlib face apis
    '''
    def __init__(self, mode="trtpose", **kwargs):
        super(DlibMatcher, self).__init__(**kwargs)
        self.mode = mode
        pre_path = "gesture_lib/data/shape_predictor_68_face_landmarks.dat"
        rec_path = "gesture_lib/data/dlib_face_recognition_resnet_model_v1.dat"
        self.face_pre = dlib.shape_predictor(pre_path)
        self.face_rec = dlib.face_recognition_model_v1(rec_path)

    def extract_feature(self, img_bgr, keypoint):
        feature = np.zeros([0, 128])

        # get face keypoints
        # Nose, REye, LEye, REar, LEar
        if self.mode == "openpose":
            keypoint = keypoint[[0, 15, 16, 17, 18], :2]
        elif self.mode == "trtpose":
            keypoint = keypoint[[0, 2, 1, 4, 3], :]
        else:
            raise

        # face keypoints missing
        if keypoint[0, 0] * keypoint[1, 0] * keypoint[2, 0] == 0:
            return feature

        # not a frontal face
        if keypoint[0, 0] < keypoint[1, 0] or keypoint[0, 0] > keypoint[2, 0]:
            return feature

        height, width = img_bgr.shape[:2]
        # face area
        x1 = keypoint[3, 0] if keypoint[3, 0] > 0 else keypoint[1, 0]*0.98
        x2 = keypoint[4, 0] if keypoint[4, 0] > 0 else keypoint[2, 0]*1.02
        x1, x2 = max(0, int(x1)), min(width, int(x2))
        w, h = x2 - x1, int((x2 - x1) * 1.1)
        y1 = keypoint[0, 1] - h / 2
        y2 = keypoint[0, 1] + h / 2
        y1, y2 = max(0, int(y1)), min(height, int(y2))
        face_img = img_bgr[y1:y2, x1:x2]

        if face_img.shape[0] * face_img.shape[1] > 0:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
            det = dlib.rectangle(0, 0, w, h)
            feature, chip = face_feature(self.face_pre, self.face_rec, face_img, det)
        return feature

    def feature_similarity(self, feature1, feature2):
        sim = 0
        if feature1.shape[0] == 0 or feature2.shape[0] == 0:
            return sim
        sim = 1 - euclidean_dis(feature1, feature2)
        return sim
