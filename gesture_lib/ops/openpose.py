import numpy as np
import cv2
import pyopenpose as op


class OpenPose(object):
    '''ops of openpose for body-25 pose estimation
    '''

    def __init__(self, model_folder, model_pose='BODY_25'):
        params = dict()
        params["model_folder"] = model_folder
        params["model_pose"] = model_pose
        self.op_wrapper = op.WrapperPython()
        self.op_wrapper.configure(params)
        self.datum = op.Datum()
        self.start = False

    def start_wrapper(self):
        if not self.start:
            self.op_wrapper.start()
            self.start = True

    def close_wrapper(self):
        if self.start:
            self.op_wrapper.stop()
            self.start = False

    def __del__(self):
        self.close_wrapper()

    def body_keypoints(self, img, resize=(224, 224)):
        '''detect body keypoints

        Args:
            op_wrapper: [op.WrapperPython()], body keypoints detector
            datum: [op.Datum()], saveing input and result
            img: [ndarray] image array
            out_img: [None | str], output result path

        Return:
            keypoints array, shape (N, point_num, 3) [x, y, score]
                point_num could be 18 or 25
            cvOutputData: detect result
        '''
        h, w = img.shape[:2]
        img_resize = cv2.resize(img, resize)
        self.start_wrapper()
        self.datum.cvInputData = img_resize
        self.op_wrapper.emplaceAndPop([self.datum])
        keypoints = self.datum.poseKeypoints
        cv_output = self.datum.cvOutputData
        cv_output = cv2.resize(cv_output, (w, h))

        if len(keypoints.shape) == 0:
            keypoints = np.zeros([0, 25, 3])

        if keypoints.shape[0] > 0:
            keypoints[:, :, 0] = keypoints[:, :, 0] * w / resize[0]
            keypoints[:, :, 1] = keypoints[:, :, 1] * w / resize[1]

        return keypoints, cv_output

    def get_best_keypoint(self, keypoints):
        '''get the body keypoints with maximum score in one frame

        for keypoints extraction

        Args:
            keypoints: [ndarray], shape (N, point_num, 3)

        Return:
            keypoint: [ndarray], shape (point_num, 3)
        '''

        scores_array = np.mean(keypoints[:, :, -1], axis=-1)
        best_idx = np.argmax(scores_array)
        return keypoints[best_idx, :, :]
