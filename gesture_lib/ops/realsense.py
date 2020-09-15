import pyrealsense2 as rs
import numpy as np
import cv2
import os
import time
from math import tan, pi
from threading import Lock
from abc import abstractmethod


class RsCamera(object):

    @staticmethod
    def show_available_device():
        ctx = rs.context()
        for i, d in enumerate(ctx.devices):
            name = d.get_info(rs.camera_info.name)
            serial_number = d.get_info(rs.camera_info.serial_number)
            print("camera id: {}, name: {}, serial number: {}"
                  .format(i, name, serial_number))

    def __init__(self, serial_number=None):
        ctx = rs.context()
        self.serial_number = serial_number
        self.rs_pipe = rs.pipeline(ctx)
        self.rs_cfg = rs.config()
        if serial_number is not None:
            self.rs_cfg.enable_device(serial_number)

    @abstractmethod
    def next_frame(self):
        raise NotImplementedError

    def start_rs_pipe(self):
        return self.rs_pipe.start(self.rs_cfg)

    def close_rs_pipe(self):
        self.rs_pipe.stop()


class RsRGBD(RsCamera):
    '''ops for Realsense D435 camera
    '''
    def __init__(self, width=640, height=480, fps=30, **kwargs):
        super(RsRGBD, self).__init__(**kwargs)
        self.rs_cfg.enable_stream(rs.stream.depth, width, height,
                                  rs.format.z16, fps)
        self.rs_cfg.enable_stream(rs.stream.color, width, height,
                                  rs.format.rgb8, fps)
        self.rs_align = rs.align(rs.stream.color)
        self.colorizer = rs.colorizer()

    def _get_aligned_frame(self, frames):
        aligned_frames = self.rs_align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        return depth_frame, color_frame

    def next_frame(self, timeout_ms=1000):
        try:
            frame = self.rs_pipe.wait_for_frames(timeout_ms=timeout_ms)
        except RuntimeError:
            frame = None
            return frame
        frame_num = frame.get_frame_number()
        depth_frame, color_frame = self._get_aligned_frame(frame)
        frame = {"color": color_frame,
                 "depth": depth_frame,
                 "number": frame_num}
        return frame

    def get_color_img(self, frame):
        if not frame:
            return None
        color_image = np.asanyarray(frame['color'].get_data())
        return color_image[:, :, ::-1]  # rgb -> bgr

    def get_depth_img(self, frame):
        if not frame:
            return None
        depth_color_frame = self.colorizer.colorize(frame['depth'])
        depth_color_image = np.asanyarray(depth_color_frame.get_data())
        return depth_color_image

    def run_stream(self):
        try:
            self.rs_pipe.start(self.rs_cfg)
        except RuntimeError:
            print("no device found for camera serial: ", self.serial_number)
            print("available realsense cameras:")
            self.show_available_device()
            exit(0)

        try:
            while True:
                frame = self.next_frame()
                if frame is None:
                    print("camera connection interrupted!")
                    break
                color_image = self.get_color_img(frame)
                depth_image = self.get_depth_img(frame)
                output = np.hstack((color_image, depth_image))
                cv2.imshow("output", output)
                cv2.waitKey(1)
        finally:
            self.rs_pipe.stop()

    def run_bag(self, bag_file, show=False):
        assert os.path.exists(bag_file)
        self.rs_cfg.enable_device_from_file(bag_file, repeat_playback=False)

        device = self.start_rs_pipe().get_device()
        play_back = rs.playback(device)
        play_back.set_real_time(False)

        try:
            while True:
                frame = self.next_frame()
                if frame is None:
                    break
                color_image = self.get_color_img(frame)
                depth_image = self.get_depth_img(frame)
                output = np.hstack((color_image, depth_image))
                if show is True:
                    cv2.imshow("output", output)
                    cv2.waitKey(1)
        except RuntimeError:
            print("camera connection interrupted!")
        finally:
            cv2.destroyAllWindows()
            self.rs_pipe.stop()


class RsFishEye(RsCamera):
    '''ops for realsense T265 fisheye camera
    '''
    def __init__(self, **kwargs):
        super(RsFishEye, self).__init__(**kwargs)
        self.min_disp = 0
        self.num_disp = 112 - self.min_disp
        self.max_disp = self.min_disp + self.num_disp
        self.frame_data = {"left": None,
                           "right": None,
                           "timestamp_ms": None}
        self.current_ts = None
        self.frame_mutex = Lock()
        self.stereo = cv2.StereoSGBM_create(minDisparity=self.min_disp,
                                            numDisparities=self.num_disp,
                                            blockSize=16,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32)
        self.fov_rad = 90 * (pi/180)
        self.undistort_rectify = {}

    def _callback(self, frame):
        if frame.is_frameset():
            frameset = frame.as_frameset()
            f1 = frameset.get_fisheye_frame(1).as_video_frame()
            f2 = frameset.get_fisheye_frame(2).as_video_frame()
            left_data = np.asanyarray(f1.get_data())
            right_data = np.asanyarray(f2.get_data())
            ts = frameset.get_timestamp()
            self.frame_mutex.acquire()
            self.frame_data["left"] = left_data
            self.frame_data["right"] = right_data
            self.frame_data["timestamp_ms"] = ts
            self.frame_mutex.release()

    def _camera_matrix(self, intrinsics):
        return np.array([[intrinsics.fx,             0, intrinsics.ppx],
                         [            0, intrinsics.fy, intrinsics.ppy],
                         [            0,             0,              1]])

    def _fisheye_distortion(self, intrinsics):
        return np.array(intrinsics.coeffs[:4])

    def _get_extrinsics(self, src, dst):
        extrinsics = src.get_extrinsics_to(dst)
        R = np.reshape(extrinsics.rotation, [3, 3]).T
        T = np.array(extrinsics.translation)
        return (R, T)

    def start_rs_pipe(self):
        return self.rs_pipe.start(self.rs_cfg, self._callback)

    def init_undistort_rectify(self, height):
        profiles = self.rs_pipe.get_active_profile()
        l_stream = profiles.get_stream(rs.stream.fisheye, 1)
        l_stream = l_stream.as_video_stream_profile()
        r_stream = profiles.get_stream(rs.stream.fisheye, 2)
        r_stream = r_stream.as_video_stream_profile()
        streams = {"left": l_stream, "right": r_stream}
        intrinsics = {"left": streams["left"].get_intrinsics(),
                      "right": streams["right"].get_intrinsics()}
        focal_px = height / 2 / tan(self.fov_rad/2)
        R, T = self._get_extrinsics(streams["left"], streams["right"])
        R_left = np.eye(3)
        R_right = R
        width = height + self.max_disp
        stereo_size = (width, height)
        stereo_cx = (height - 1)/2 + self.max_disp
        stereo_cy = (height - 1)/2
        P_left = np.array([[focal_px, 0, stereo_cx, 0],
                           [0, focal_px, stereo_cy, 0],
                           [0,        0,         1, 0]])
        P_right = P_left.copy()

        K_left = self._camera_matrix(intrinsics["left"])
        D_left = self._fisheye_distortion(intrinsics["left"])
        K_right = self._camera_matrix(intrinsics["right"])
        D_right = self._fisheye_distortion(intrinsics["right"])

        # Get the relative extrinsics between the left and right camera
        P_right[0][3] = T[0]*focal_px

        lm1, lm2 = cv2.fisheye.initUndistortRectifyMap(K_left,
                                                       D_left,
                                                       R_left,
                                                       P_left,
                                                       stereo_size,
                                                       cv2.CV_32FC1)
        rm1, rm2 = cv2.fisheye.initUndistortRectifyMap(K_right,
                                                       D_right,
                                                       R_right,
                                                       P_right,
                                                       stereo_size,
                                                       cv2.CV_32FC1)

        self.undistort_rectify["left"] = (lm1, lm2)
        self.undistort_rectify["right"] = (rm1, rm2)

    def next_frame(self, timeout_ms=1000):
        assert self.undistort_rectify != {}, \
            "please init the undistort_rectify before start!"
        t_s = time.time()
        while True:
            self.frame_mutex.acquire()
            valid = self.frame_data["timestamp_ms"] != self.current_ts
            if (time.time() - t_s) * 1000 >= timeout_ms:
                return None
            if not valid:
                self.frame_mutex.release()
                continue
            else:
                self.current_ts = self.frame_data["timestamp_ms"]
                frame_copy = {"left": self.frame_data["left"].copy(),
                              "right": self.frame_data["right"].copy()}
                self.frame_mutex.release()
                break

        # Undistort and crop the center of the frames
        center_left = cv2.remap(src=frame_copy["left"],
                                map1=self.undistort_rectify["left"][0],
                                map2=self.undistort_rectify["left"][1],
                                interpolation=cv2.INTER_LINEAR)
        center_right = cv2.remap(src=frame_copy["right"],
                                 map1=self.undistort_rectify["right"][0],
                                 map2=self.undistort_rectify["right"][1],
                                 interpolation=cv2.INTER_LINEAR)
        center_undistorted = {"left": center_left,
                              "right": center_right}
        return center_undistorted

    def get_color_img(self, frame, camera):
        fish_img = frame[camera][:, self.max_disp:]
        fish_img = cv2.cvtColor(fish_img, cv2.COLOR_GRAY2RGB)
        return fish_img

    def run(self, height=300, camera="left"):
        assert camera in ("left", "right", "both")
        try:
            self.start_rs_pipe()
        except RuntimeError:
            print("no device found for camera serial: ", self.serial_number)
            print("available realsense cameras:")
            self.show_available_device()
            exit(0)
        try:
            self.init_undistort_rectify(height)
            while True:
                # If frames are ready to process
                frame = self.next_frame()
                if frame is None:
                    print("camera connection interrupted!")
                    break
                if camera in ("left", "right"):
                    color_image = self.get_color_img(frame, camera)
                else:
                    l_image = self.get_color_img(frame, "left")
                    r_image = self.get_color_img(frame, "right")
                    color_image = np.hstack((l_image, r_image))
                # print(color_image.shape)
                color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
                cv2.imshow("color image", color_image)
                cv2.waitKey(1)
        except RuntimeError:
            print("camera connection interrupted!")
        finally:
            self.close_rs_pipe()
