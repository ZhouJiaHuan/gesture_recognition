import time
import numpy as np
import cv2

from gesture_lib.datasets import Body3DExtractor

from gesture_lib.ops.realsense import RsRGBD, RsFishEye
from gesture_lib.ops.yaml_utils import parse_yaml


class InferenceMulticam(object):

    def __init__(self):
        core_cfg = parse_yaml("gesture_lib/configs.yaml")

        # realsense D435 and T265
        self.width = core_cfg.width
        self.height = core_cfg.height
        self.size = min(self.width, self.height)
        self.D435 = core_cfg.camera_id.D435
        self.rgbd = RsRGBD(self.width, self.height, serial_number=self.D435)
        self.T265 = core_cfg.camera_id.T265
        self.fisheye = RsFishEye(serial_number=self.T265)

        # pose estimation
        pose_params = core_cfg.pose_params
        self.mode = pose_params.mode
        self.pose_w = pose_params.pose_w
        self.pose_h = pose_params.pose_h
        self.extractor = Body3DExtractor()

    def start_rs_pipe(self):
        try:
            self.rgbd.start_rs_pipe()
        except RuntimeError:
            print("no device found for D435 camera serial: ", self.D435)
            exit(0)
        try:
            self.fisheye.start_rs_pipe()
        except RuntimeError:
            print("no device found for T265 camera serial: ", self.T265)
            exit(0)

    def close_rs_pipe(self):
        self.rgbd.close_rs_pipe()
        self.fisheye.close_rs_pipe()

    def _body_keypoints(self, rgbd_img, fish_img):
        '''detect body keypoints from a frame

        Args:
            frame: one frame from realsense, including color
                and depth frame info

        # TODO
            improve with multi thread

        Return:
            keypoints: [ndarray], pose keypoints array,
                shape (N, keypoint_num, :), (x, y) or (x, y, score)
            cv_output: [ndarray]: output image with keypoints info.

        '''
        kp1, out1 = self.extractor.body_keypoints(rgbd_img)
        fish_img = cv2.cvtColor(fish_img, cv2.COLOR_GRAY2RGB)
        kp2, out2 = self.extractor.body_keypoints(fish_img)
        return kp1, out1, kp2, out2

    def _draw_fps(self, image, t):
        image = np.array(image, dtype=np.uint8)
        speed = str(int(1 / t)) + ' FPS'
        cv2.putText(image, speed, (image.shape[1]-120, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        return image

    def run(self, show=False, fisheye="left"):
        self.start_rs_pipe()
        self.fisheye.init_undistort_rectify(self.size)
        try:
            while True:
                rgbd_frame = self.rgbd.next_frame()
                fish_frame = self.fisheye.next_frame()
                if (rgbd_frame is None) or (fish_frame is None):
                    print("camera disconnected!")
                    print("avaliable realsense camera: ")
                    self.fisheye.show_available_device()
                    break
                time_s = time.time()
                rgbd_img = self.rgbd.get_color_img(rgbd_frame)
                fish_img = self.fisheye.get_color_img(fish_frame, fisheye)
                kp1, out1, kp2, out2 = self._body_keypoints(rgbd_img, fish_img)
                time_e = time.time()

                if show:
                    out1 = self._draw_fps(out1, time_e-time_s)
                    out2 = self._draw_fps(out2, time_e-time_s)
                    cv2.imshow("rgbd color", out1)
                    cv2.imshow("fish color", out2)
                    cv2.waitKey(1)
        except RuntimeError:
            print("camera disconnected!")
        finally:
            self.close_rs_pipe()

