import os
import cv2
import numpy as np
import pyrealsense2 as rs
from tqdm import tqdm

from gesture_lib.ops.realsense import RsRGBD
from gesture_lib.ops.io import get_file_path
from gesture_lib.ops.yaml_utils import parse_yaml


class Body3DExtractor(object):
    '''3-D body keypoints extractor from the frame
       collected with Realsense RGB-D camera.
    '''

    def __init__(self):
        '''
        Args:

        '''
        cfg = parse_yaml('gesture_lib/configs.yaml')
        self.width = cfg.width
        self.height = cfg.height
        self.fps = cfg.fps

        pose_params = cfg.pose_params
        self.mode = pose_params.mode
        self.pose_w = pose_params.pose_w
        self.pose_h = pose_params.pose_h
        self.extractor = self._build_pose_extractor(pose_params)


        # configure realsense
        self.rs_rgbd = RsRGBD(self.width, self.height, self.fps)

    def _build_pose_extractor(self, pose_params):
        params = {}
        if self.mode == 'trtpose':
            try:
                from gesture_lib.ops.trtpose import TrtPose
            except Exception:
                print("Trtpose library not found!")
                exit(0)
            params['trt_model'] = pose_params.trt_model
            params['pose_json_path'] = pose_params.pose_json_path
            extractor = TrtPose(**params)
        elif self.mode == 'openpose':
            try:
                from gesture_lib.ops.openpose import OpenPose
            except Exception:
                print("Openpose library not found!")
                exit(0)
            params['model_folder'] = pose_params.op_model
            params['model_pose'] = pose_params.model_pose
            extractor = OpenPose(**params)
        else:
            raise KeyError("invalid pose mode: {}".format(self.mode))
        return extractor

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

    def body_keypoints(self, img):
        '''
        Args:
            img: [ndarray] image array

        Return:
            keypoints array, shape (N, points_num, :) [pixel_x, pixel_y, ...]
                points_num could be 18 or 25. for openpose, each keypoint has
                a confidence score
            cvOutputData: detect result
        '''
        resize = (self.pose_w, self.pose_h)
        return self.extractor.body_keypoints(img, resize)

    def get_best_keypoint(self, keypoints):
        '''
        Args:
            keypoints: [ndarray], shape (N, :, :)

        Return:
            keypoint: [ndarray], shape (points_num, :) [pixel_x, pixel_y, ...]
        '''
        return self.extractor.get_best_keypoint(keypoints)

    def _op_keypoint_to_point(self, keypoint, depth_frame):
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
                continue  # ignore invalid keypoint
            else:
                temp_point = self.pixel_to_point(depth_frame, temp_keypoint[:2])
            point[i, :3] = temp_point

        return point

    def _trt_keypoint_to_point(self, keypoint, depth_frame):
        '''
        Args:
            keypoint: [ndarray], shape (points_num, 2) [pixel_x, pixel_y]
            depth_frame: depth frame from realsense

        Return:
            point: [ndarray], shape (points_num, 3) [point_x, point_y, point_z]
        '''
        point_num = keypoint.shape[0]
        point = np.zeros([point_num, 3])
        for i in range(point_num):
            if keypoint[i, 0] < 1e-3 or keypoint[i, 1] < 1e-3:
                continue
            point[i, :3] = self.pixel_to_point(depth_frame, keypoint[i, ])

        return point

    def keypoint_to_point(self, keypoint, depth_frame):
        '''get the point (3-D location) with keypoint info and depth frame
        '''
        if self.mode == 'trtpose':
            return self._trt_keypoint_to_point(keypoint, depth_frame)
        elif self.mode == 'openpose':
            return self._op_keypoint_to_point(keypoint, depth_frame)
        else:
            raise KeyError("invalid pose mode: {}".format(self.mode))

    def process_bag(self, bag_path, show=False):
        '''parse a bag file

        Args:
            bag_path: [str] bag file path

        Return:
            points_array: [ndarray], 3-D body keypoints info, (T, :)
        '''
        assert os.path.exists(bag_path)
        self.rs_rgbd.rs_cfg.enable_device_from_file(bag_path, repeat_playback=False)

        device = self.rs_rgbd.start_rs_pipe().get_device()
        play_back = rs.playback(device)
        play_back.set_real_time(False)

        points_array = []
        frame_number = 0
        try:
            while True:
                frame = self.rs_rgbd.next_frame()
                if not frame:
                    break
                current_frame_num = frame['number']
                if current_frame_num == frame_number:
                    continue
                else:
                    frame_number = current_frame_num
                color_image = self.rs_rgbd.get_color_img(frame)
                depth_frame = frame['depth']

                keypoints, cv_output = self.body_keypoints(color_image)
                if show is True:
                    cv2.imshow("keypoints output", cv_output)
                    cv2.waitKey(1)
                if keypoints.shape[0] < 1:
                    continue

                keypoint = self.get_best_keypoint(keypoints)
                point = self.keypoint_to_point(keypoint, depth_frame)
                points_array.append(point.flatten())
        finally:
            cv2.destroyAllWindows()
            self.rs_rgbd.close_rs_pipe()

        points_array = np.array(points_array, dtype='float')
        return points_array

    def extract(self, bag_dir, save_ext='.txt', show=False):
        '''
        TODO
            1. multi-thread
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
