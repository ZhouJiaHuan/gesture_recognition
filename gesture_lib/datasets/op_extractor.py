import cv2
import numpy as np
import pyopenpose as op
from .body_extractor import BodyExtractor


class OpenposeExtractor(BodyExtractor):
    '''
    '''

    def __init__(self,
                 pose_model_folder,
                 model_pose,
                 *args,
                 **kwargs):
        super(OpenposeExtractor, self).__init__(*args, **kwargs)
        self.pose_model_folder = pose_model_folder
        self.model_pose = model_pose
        self.op_wrapper, self.datum = self._config_op()
        self.op_wrapper.start()

    def __del__(self):
        self.op_wrapper.stop()

    def _config_op(self):
        params = dict()
        params["model_folder"] = self.pose_model_folder
        params["model_pose"] = self.model_pose
        op_wrapper = op.WrapperPython()
        op_wrapper.configure(params)
        datum = op.Datum()
        return op_wrapper, datum

    def body_keypoints(self, img_process):
        '''detect body keypoints

        Args:
            op_wrapper: [op.WrapperPython()], body keypoints detector
            datum: [op.Datum()], saveing input and result
            img_process: [ndarray] image array
            out_img: [None | str], output result path

        Return:
            keypoints array, shape (N, point_num, 3) [x, y, score]
                point_num could be 18 or 25
            cvOutputData: detect result
        '''
        self.datum.cvInputData = img_process
        self.op_wrapper.emplaceAndPop([self.datum])
        keypoints = self.datum.poseKeypoints
        if len(keypoints.shape) == 0:
            keypoints = np.zeros([0, 25, 3])

        return keypoints, self.datum.cvOutputData

    def get_best_keypoint(self, keypoints):
        '''get the body keypoints with maximum score in one frame

        for keypoints extraction

        Args:
            keypoints: [ndarray], shape (N, point_num, 3)

        Return:
            keypoint: [ndarray], shape (point_num, 3)
        '''

        scores_array = np.mean(keypoints[:,:,-1], axis=-1)
        best_idx = np.argmax(scores_array)
        return keypoints[best_idx, :, :]

    def keypoint_to_point(self, keypoint, depth_frame):
        '''
        Args:
            keypoint: [ndarray], shape (points_num, 3) [pixel_x, pixel_y, score]
            depth_frame: depth frame from realsense

        Return:
            point: [ndarray], shape (points_num, 4) [point_x, point_y, point_z, score]
        '''
        point_num = keypoint.shape[0]
        point = np.zeros([point_num, 4])
        point[:, -1] = keypoint[:, -1]
        for i in range(point_num):
            temp_keypoint = keypoint[i, ]
            if temp_keypoint[-1] < 1e-3:
                temp_point = np.array([0, 0, 0])
            else:
                temp_point = self.pixel_to_point(depth_frame, temp_keypoint[:2])
            point[i, :3] = temp_point

        return point
