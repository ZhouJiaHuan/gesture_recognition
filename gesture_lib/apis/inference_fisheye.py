import time
import numpy as np
import cv2

from gesture_lib.datasets import Body3DExtractor

from gesture_lib.ops.realsense import RsFishEye
from gesture_lib.ops.yaml_utils import parse_yaml


class InferenceFisheye(object):

    def __init__(self):
        core_cfg = parse_yaml("gesture_lib/configs.yaml")

        # realsense T265
        self.T265 = core_cfg.camera_id.T265
        self.fisheye = RsFishEye(serial_number=self.T265)
        self.width = core_cfg.width
        self.height = core_cfg.height
        self.size = min(self.width, self.height)

        pose_params = core_cfg.pose_params
        self.mode = pose_params.mode
        self.pose_w = pose_params.pose_w
        self.pose_h = pose_params.pose_h
        self.extractor = Body3DExtractor()

    def _draw_fps(self, image, t):
        image = np.array(image, dtype=np.uint8)
        speed = str(int(1 / t)) + ' FPS'
        cv2.putText(image, speed, (image.shape[1]-120, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return image

    def _body_keypoints(self, img):
        '''detect body keypoints from a frame

        Args:
            frame: one frame from realsense, including color
                and depth frame info

        Return:
            keypoints: [ndarray], pose keypoints array,
                shape (N, keypoint_num, :), (x, y) or (x, y, score)
            cv_output: [ndarray]: output image with keypoints info.

        '''
        return self.extractor.body_keypoints(img)

    def run(self, camera="left", show=False):
        assert camera in ("left", "right")
        try:
            self.fisheye.start_rs_pipe()
        except RuntimeError:
            print("no device found for camera serial: ", self.T265)
            print("avaliable realsense camera: ")
            self.fisheye.show_available_device()
            exit(0)
        try:
            self.fisheye.init_undistort_rectify(self.size)
            while True:
                frame = self.fisheye.next_frame()
                if frame is None:
                    print("camera connection interrupted!")
                    print("avaliable realsense camera: ")
                    self.fisheye.show_available_device()
                    break
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
