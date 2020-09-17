from multiprocessing import Queue
import cv2
import time
import numpy as np

from gesture_lib.datasets import Body3DExtractor

from gesture_lib.ops.realsense import RsFishEye
from gesture_lib.ops.yaml_utils import parse_yaml

camera = "left"
core_cfg = parse_yaml("gesture_lib/configs.yaml")
T265 = core_cfg.camera_id.T265
fisheye = RsFishEye(serial_number=T265)

width = core_cfg.width
height = core_cfg.height
size = min(width, height)


def draw_fps(image, t):
    image = np.array(image, dtype=np.uint8)
    speed = str(int(1 / t)) + ' FPS'
    cv2.putText(image, speed, (image.shape[1]-120, 50),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
    return image


def producer(q: Queue):
    try:
        fisheye.start_rs_pipe()
    except RuntimeError:
        print("no device found for camera serial: ", T265)
        print("avaliable realsense camera: ")
        fisheye.show_available_device()
        exit(0)
    try:
        fisheye.init_undistort_rectify(size)
        while True:
            frame = fisheye.next_frame(timeout_ms=1000)
            if frame is None:
                print("camera connection interrupted!")
                print("avaliable realsense camera: ")
                fisheye.show_available_device()
                break
            q.put(frame)
    finally:
        fisheye.close_rs_pipe()


def consumer(q: Queue, show=True):
    extractor = Body3DExtractor()
    while True:
        frame = q.get()
        if frame is None:
            print("no frame data available!")
            break
        time_s = time.time()
        img = fisheye.get_color_img(frame, camera)
        keypoints, cv_output = extractor.body_keypoints(img)
        time_e = time.time()
        if show:
            cv_output = draw_fps(cv_output, time_e-time_s)
            cv2.imshow("detect result", cv_output)
            cv2.waitKey(1)
    cv2.destroyAllWindows()
