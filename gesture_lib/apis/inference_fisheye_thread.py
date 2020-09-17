import time
import numpy as np
import cv2
import threading
from collections import deque

from gesture_lib.datasets import Body3DExtractor

from gesture_lib.ops.realsense import RsFishEye
from gesture_lib.ops.yaml_utils import parse_yaml


lock = threading.Lock()


class FisheyeStream(threading.Thread):
    def __init__(self, cache: deque, camera="left"):
        super(FisheyeStream, self).__init__()
        assert camera in ("left", "right")
        self.cache = cache
        self.max_len = cache.maxlen
        self.camera = camera
        core_cfg = parse_yaml("gesture_lib/configs.yaml")
        # realsense T265
        self.T265 = core_cfg.camera_id.T265
        self.fisheye = RsFishEye(serial_number=self.T265)
        self.width = core_cfg.width
        self.height = core_cfg.height
        self.size = min(self.width, self.height)

    def run(self):
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
                frame = self.fisheye.next_frame(timeout_ms=5000)
                if frame is None:
                    print("camera connection interrupted!")
                    print("avaliable realsense camera: ")
                    # self.fisheye.show_available_device()
                    continue
                lock.acquire()
                if len(self.cache) == self.max_len:
                    self.cache.popleft()
                else:
                    self.cache.append(frame)
                lock.release()
        finally:
            self.cache.clear()
            self.fisheye.close_rs_pipe()


class PoseEst(threading.Thread):
    def __init__(self, cache: deque, camera="left", show=True):
        super(PoseEst, self).__init__()
        self.cache = cache
        self.camera = camera
        core_cfg = parse_yaml("gesture_lib/configs.yaml")
        self.T265 = core_cfg.camera_id.T265
        self.fisheye = RsFishEye(serial_number=self.T265)
        self.extractor = Body3DExtractor()
        self.show = show

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

    def run(self):
        while True:
            if len(self.cache) > 0:
                lock.acquire()
                frame = self.cache.pop()
                # if frame is None:
                #     continue
                time_s = time.time()
                img = self.fisheye.get_color_img(frame, self.camera)
                lock.release()
                keypoints, cv_output = self._body_keypoints(img)
                time_e = time.time()
                if self.show:
                    cv_output = self._draw_fps(cv_output, time_e-time_s)
                    cv2.imshow("detect result", cv_output)
                    cv2.waitKey(1)
