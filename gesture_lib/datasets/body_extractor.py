
import os
import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm
from abc import abstractmethod

from gesture_lib.ops import get_file_path


class BodyExtractor(object):
    '''extract 3-D locations of body keypoints from .bag file.
    '''

    def __init__(self,
                 width=640,
                 height=480,
                 fps=30,
                 depth_format=rs.format.z16,
                 color_format=rs.format.rgb8):
        '''
        Args:

        '''
        self.width = width
        self.height = height
        self.fps = fps
        self.depth_format = depth_format
        self.color_format = color_format

        # configure realsense
        self.rs_pipe, self.rs_cfg = self._config_rs()

    def _config_rs(self):
        rs_pipe = rs.pipeline()
        rs_cfg = rs.config()   
        rs_cfg.enable_stream(rs.stream.depth, self.width, self.height, self.depth_format, self.fps)
        rs_cfg.enable_stream(rs.stream.color, self.width, self.height, self.color_format, self.fps)
        return rs_pipe, rs_cfg

    def _get_aligned_frame(self, frames, with_color=True):
        if with_color is True:
            align = rs.align(rs.stream.color)
        else:
            align = rs.align(rs.stream.depth)
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return depth_frame, color_frame

    def pixel_to_point(self, depth_frame, xy):
        '''get the point from depth frame
        '''
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        x, y = xy
        x = max(min(x, width-1), 0)
        y = max(min(y, height-1), 0)
        x1, x2 = int(np.floor(x)), int(np.ceil(x))
        y1, y2 = int(np.floor(y)), int(np.ceil(y))

        depth_intrinsics = rs.video_stream_profile(depth_frame.profile).get_intrinsics()
        if x1 == x2 and y1 == y2:
            dist = depth_frame.get_distance(x1, y1)
            point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x, y], dist)
        else:
            dist11 = depth_frame.get_distance(x1, y1)
            point11 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x1, y1], dist11)
            dist12 = depth_frame.get_distance(x1, y2)
            point12 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x1, y2], dist12)
            dist21 = depth_frame.get_distance(x2, y1)
            point21 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x2, y1], dist21)
            dist22 = depth_frame.get_distance(x2, y2)
            point22 = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [x2, y2], dist22)
            point = (x2-x)*(y2-y) * np.array(point11) + (x2-x)*(y-y1) * np.array(point12) + \
                    (x-x1)*(y2-y) * np.array(point21) + (x-x1)*(y-y1) * np.array(point22)

        return point

    @abstractmethod
    def body_keypoints(self, img_process):
        '''
        Args:
            img_process: [ndarray] image array

        Return:
            keypoints array, shape (N, points_num, :) [pixel_x, pixel_y, ...]
                points_num could be 18 or 25. for openpose, each keypoint has
                a confidence score
            cvOutputData: detect result
        '''
        raise NotImplementedError

    @abstractmethod
    def get_best_keypoint(self, keypoints):
        '''
        Args:
            keypoints: [ndarray], shape (N, :, :)

        Return:
            keypoint: [ndarray], shape (points_num, :) [pixel_x, pixel_y, ...]
        '''
        raise NotImplementedError

    @abstractmethod
    def keypoint_to_point(self, keypoint, depth_frame):
        '''get the point (3-D location) with keypoint info and depth frame

        Args:
            keypoint: [ndarray], shape (points_num, :) [pixel_x, pixel_y, ...]
            depth_frame: depth frame from realsense

        Return:
            point: [ndarray], shape (points_num, :) [point_x, point_y, point_z, ...]
        '''
        raise NotImplementedError

    def process_bag(self, bag_path, show=False):
        '''parse a bag file

        Args:
            bag_path: [str] bag file path

        Return:
            points_array: [ndarray], 3-D body keypoints info, (T, :)
        '''
        assert os.path.exists(bag_path)
        self.rs_cfg.enable_device_from_file(bag_path, repeat_playback=False)

        device = self.rs_pipe.start(self.rs_cfg).get_device()
        play_back = rs.playback(device)
        play_back.set_real_time(False)

        points_array = []
        frame_number = 0
        try:
            while True:
                frames = self.rs_pipe.wait_for_frames(timeout_ms=1000)
                current_frame_num = frames.get_frame_number()
                if current_frame_num == frame_number:
                    continue
                else:
                    frame_number = current_frame_num

                depth_frame, color_frame = self._get_aligned_frame(frames)

                if (not depth_frame) or (not color_frame):
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                keypoints, cv_output = self.body_keypoints(color_image)
                if show is True:
                    cv_output = cv_output[:, :, ::-1]
                    cv2.imshow("keypoints output", cv_output)
                    cv2.waitKey(1)
                if keypoints.shape[0] < 1:
                    continue

                keypoint = self.get_best_keypoint(keypoints)
                point = self.keypoint_to_point(keypoint, depth_frame)
                points_array.append(point.flatten())
        except RuntimeError:
            cv2.destroyAllWindows()
            self.rs_pipe.stop()

        points_array = np.array(points_array, dtype='float')
        return points_array

    def extract(self, bag_dir, save_ext='.txt', show=False):
        '''
        Args:
            bag_dir: [str], bag files directory
        '''
        bag_list = get_file_path(bag_dir, filter='.bag')

        for bag_path in tqdm(bag_list):
            try:
                points_array = self.process_bag(bag_path, show)
            except Exception as e:
                print("parse failed for: {}".format(bag_path))
                print(e)
                continue
            try:
                txt_path = bag_path.replace('.bag', save_ext)
                np.savetxt(txt_path, points_array, fmt='%.5f')
            except Exception as e:
                print("save failed for: {}".format(bag_path))
                print(e)
                continue
