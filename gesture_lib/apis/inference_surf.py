import sys
sys.path.append(".")

import time
import numpy as np
import cv2

from .inference import Inference


class InferenceSurf(Inference):
    '''inference of body keypoints model for video from realsense
    '''

    def __init__(self, *args, **kwargs):
        super(InferenceSurf, self).__init__(*args, **kwargs)
        self.face_feature = cv2.xfeatures2d.SURF_create(200, extended=False, upright=0)
        self.body_feature = cv2.xfeatures2d.SURF_create(100, extended=False, upright=0)
        index_param = dict(algorithm=0, trees=5)
        search_param = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_param, search_param)

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

    def _feature_similarity(self, feature1, feature2):
        similarity = 0
        if feature1.shape[0] < 2 or feature2.shape[0] < 2:
            return similarity
        feature1 = np.asarray(feature1, dtype=np.float32)
        feature2 = np.asarray(feature2, dtype=np.float32)
        matches = self.matcher.knnMatch(feature2, feature1, k=2)

        num = 0
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                num += 1
        similarity = num / len(matches)
        return similarity

    def _person_sim(self, person_1, person_2):
        face_feature1, body_feature1 = person_1
        face_feature2, body_feature2 = person_2
        # print(face_feature1.shape, face_feature2.shape)
        # print(body_feature1.shape, body_feature2.shape)
        face_sim = self._feature_similarity(face_feature1, face_feature2)
        body_sim = self._feature_similarity(body_feature1, body_feature2)
        sim = 1 * face_sim + 0.0 * body_sim
        # print("face sim: {:.3f}, body sim: {:.3f}, total sim: {:.3f}".format(face_sim, body_sim, sim))

        return sim

    def _is_in_memory(self, sim, sim_thr=0.2):
        return sim > sim_thr

    def _update_memory(self, color_image, keypoints_list, points_list):
        result_ids = []
        # print("keypoints num: {}".format(len(keypoints_list)))
        for keypoint, point in zip(keypoints_list, points_list):
            face_feature = self._get_face_feature(color_image, keypoint)
            body_feature = self._get_body_feature(color_image, keypoint)
            if face_feature.shape[0] == 0 or body_feature.shape[0] == 0:
                continue

            input_info = [face_feature, body_feature]
            best_person_id, best_sim = self._find_best_match(input_info)

            if best_person_id in result_ids:
                continue

            if self._is_in_memory(best_sim):
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
