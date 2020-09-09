import time
import numpy as np
import cv2

from gesture_lib.datasets import TrtposeExtractor
try:
    from gesture_lib.datasets import OpenposeExtractor
except Exception:
    print("openpose library not found!")

from gesture_lib.ops.fisheye import RsFishEye
from gesture_lib.ops.yaml_utils import parse_yaml


class InferenceFisheye(object):

    def __init__(self):
        infer_cfg = parse_yaml("gesture_lib/apis/inference.yaml")

        # realsense frame size
        self.width = infer_cfg.width
        self.height = infer_cfg.height
        self.fisheye = RsFishEye()

        self.op_wrapper = None
        pose_params = infer_cfg.pose_params
        self._configure_gesture_est(pose_params)

        # configure realsense camera
        self.rs_pipe = self.extractor.rs_pipe
        self.rs_cfg = self.extractor.rs_cfg

    def _configure_gesture_est(self, pose_params):
        self.mode = pose_params['mode']
        self.pose_w = pose_params.pose_w
        self.pose_h = pose_params.pose_h
        if self.mode == "openpose":
            op_model = pose_params['op_model']
            model_pose = pose_params['model_pose']
            self.extractor = OpenposeExtractor(op_model, model_pose)
            self.pose_wscale = self.width / self.pose_w
            self.pose_hscale = self.height / self.pose_h
            self.dim = 4  # (x, y, z, score)
            self.op_wrapper = self.extractor.op_wrapper
            self.datum = self.extractor.datum
        elif self.mode == "trtpose":
            trt_model = pose_params['trt_model']
            pose_json_path = pose_params['pose_json_path']
            self.extractor = TrtposeExtractor(trt_model,
                                              pose_json_path)
            self.pose_wscale = 1
            self.pose_hscale = 1
            self.dim = 3  # (x, y, z)
        else:
            exit(0)

    def _draw_fps(self, image, t):
        image = np.array(image, dtype=np.uint8)
        speed = str(int(1 / t)) + ' FPS'
        cv2.putText(image, speed, (image.shape[1]-120, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return image

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

    def run(self, camera="left", show=False):
        assert camera in ("left", "right")
        self.fisheye.init_rs_pipe()
        try:
            self.fisheye.init_undistort_rectify(self.height)
            while True:
                # If frames are ready to process
                if self.fisheye.is_valid_frame():
                    frame = self.fisheye.next_frame()
                    time_s = time.time()
                    color_image = self.fisheye.get_color_img(frame, camera)
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
                    keypoints, cv_output = self._body_keypoints(color_image)
                    time_e = time.time()
                    if show:
                        cv_output = self._draw_fps(cv_output, time_e-time_s)
                        cv2.imshow("detect result", cv_output)
                        cv2.waitKey(1)
        finally:
            self.fisheye.close_rs_pipe()
