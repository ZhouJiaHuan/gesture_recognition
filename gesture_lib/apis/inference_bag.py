import time
import os
import cv2
import pyrealsense2 as rs

from gesture_lib.ops.realsense import RsRGBD
from .inference_rgbd import InferenceRGBD


class InferenceBag(InferenceRGBD):
    ''' inference of body keypoints model for RGB-D bag file

    mainly for testing of RGB-D realsense inference without D435 camera

    '''

    def __init__(self, cfg_path, checkpoints):
        super(InferenceBag, self).__init__(cfg_path, checkpoints)
        assert os.path.splitext(checkpoints)[-1] == '.pth'

        self.rs_camera = RsRGBD()

    def run(self, bag_path, show=False):
        assert os.path.exists(bag_path)
        try:
            self.rs_camera.rs_cfg.enable_device_from_file(bag_path, repeat_playback=False)
            device = self.rs_camera.start_rs_pipe().get_device()
            # play_back = rs.playback(device)
            # play_back.set_real_time(False)
        except Exception as e:
            print(e)
            exit(0)

        frame_number = 0
        while True:
            print('------------------------------')
            frame = self.rs_camera.next_frame()

            if not frame:
                break
            color_image = self.rs_camera.get_color_img(frame)
            depth_frame = frame['depth']
            if frame['number'] == frame_number:
                continue
            frame_number = frame['number']
            time_s = time.time()
            skeletons, cv_output = self._detect_frame(color_image, depth_frame)
            # target_person = self.target['person_id']
            self._multi_track(color_image, skeletons)
            # if target_person == '0':
            #     self._multi_track(color_image, skeletons)
            # else:
            #     self._single_track(color_image, skeletons)
            time_e = time.time()
            if show is True:
                cv_output = self._draw_person_ids(cv_output, skeletons)
                cv_output = self._draw_fps(cv_output, time_e-time_s)
                cv_output = self._draw_target(cv_output)
                cv2.imshow("gesture output", cv_output)
                cv2.waitKey(1)

        self.rs_camera.close_rs_pipe()
