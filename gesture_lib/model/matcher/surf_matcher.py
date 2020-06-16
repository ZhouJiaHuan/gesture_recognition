import numpy as np
import cv2
from .base_matcher import BaseMatcher


class SurfMatcher(BaseMatcher):

    def __init__(self, mode="trtpose", belta=0.5, **kwargs):
        super(SurfMatcher, self).__init__(**kwargs)
        self.belta = belta
        self.mode = mode
        self.face_feature = cv2.xfeatures2d.SURF_create(200, extended=False, upright=0)
        self.body_feature = cv2.xfeatures2d.SURF_create(100, extended=False, upright=0)
        index_param = dict(algorithm=0, trees=5)
        search_param = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_param, search_param)

    def _surf_extractor(self, extractor, img):
        kp, des = extractor.detectAndCompute(img, None)
        return kp, des

    def _get_body_feature(self, img_bgr, keypoint):
        feature = np.zeros([0, self.body_feature.descriptorSize()])
        if self.mode == "openpose":
            keypoint = keypoint[[2, 5, 8], :]
            keypoint = keypoint[keypoint[:, -1] > 0, :2]
            if keypoint.shape[0] < 3:
                return feature
        else:
            keypoint = keypoint[[6, 5, 12, 11], :]
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
        img = img_bgr[y1:y2, x1:x2]
        if img.shape[0] * img.shape[1] > 0:
            cols = x2 - x1
            rows = y2 - y1
            src = np.float32(keypoint - [x1, y1])
            dst = np.float32([[0, 0], [cols, 0], [cols/2, rows]])
            M = cv2.getAffineTransform(src, dst)
            img_affine = cv2.warpAffine(img, M, (cols, rows))
            img_affine = cv2.resize(img_affine, (80, 120))
            _, temp_feature = self._surf_extractor(self.body_feature, img_affine)
            feature = temp_feature if temp_feature is not None else feature
        return feature

    def _get_face_feature(self, img_bgr, keypoint):
        feature = np.zeros([0, self.face_feature.descriptorSize()])
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

        height, width = img_bgr.shape[:2]
        x1 = keypoint[3, 0] if keypoint[3, 0] > 0 else keypoint[1, 0]*0.98
        x2 = keypoint[4, 0] if keypoint[4, 0] > 0 else keypoint[2, 0]*1.02
        x1, x2 = max(0, int(x1)), min(width, int(x2))
        w, h = x2 - x1, int((x2 - x1) * 1.1)
        y1 = keypoint[0, 1] - h / 2
        y2 = keypoint[0, 1] + h / 2
        y1, y2 = max(0, int(y1)), min(height, int(y2))
        img = img_bgr[y1:y2, x1:x2]
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 0, 255))

        if img.shape[0] * img.shape[1] > 0:
            src = np.float32(keypoint[:3, ] - [x1, y1])
            angle = np.arctan2(src[2, 1]-src[1, 1], src[2, 0]-src[1, 0]) * 180 / np.pi
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            img_affine = cv2.warpAffine(img, M, (w, h))
            img_affine = cv2.resize(img_affine, (96, 112))
            _, temp_feature = self._surf_extractor(self.face_feature, img_affine)
            feature = temp_feature if temp_feature is not None else feature

        return feature

    def _feature_similarity(self, feature1, feature2):
        sim = 0
        if feature1.shape[0] < 2 or feature2.shape[0] < 2:
            return sim
        feature1 = np.asarray(feature1, dtype=np.float32)
        feature2 = np.asarray(feature2, dtype=np.float32)
        matches = self.matcher.knnMatch(feature2, feature1, k=2)

        num = 0
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                num += 1
        sim = num / len(matches)
        return sim

    def extract_feature(self, img_bgr, keypoint):
        face_feature = self._get_face_feature(img_bgr, keypoint)
        body_feature = self._get_body_feature(img_bgr, keypoint)
        return [face_feature, body_feature]

    def feature_similarity(self, feature1, feature2):
        face_feature1, body_feature1 = feature1
        face_feature2, body_feature2 = feature2
        face_sim = self._feature_similarity(face_feature1, face_feature2)
        body_sim = self._feature_similarity(body_feature1, body_feature2)
        sim = self.belta * face_sim + (1-self.belta) * body_sim
        return sim
