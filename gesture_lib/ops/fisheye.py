# First import the library
import pyrealsense2 as rs

# Import OpenCV and numpy
import cv2
import numpy as np
import os
from math import tan, pi
from threading import Lock

frame_mutex = Lock()


class RsFishEye(object):
    def __init__(self):
        self.min_disp = 0
        self.num_disp = 112 - self.min_disp
        self.max_disp = self.min_disp + self.num_disp
        self.frame_data = {"left": None,
                           "right": None,
                           "timestamp_ms": None}
        self.stereo = cv2.StereoSGBM_create(minDisparity=self.min_disp,
                                            numDisparities=self.num_disp,
                                            blockSize=16,
                                            disp12MaxDiff=1,
                                            uniquenessRatio=10,
                                            speckleWindowSize=100,
                                            speckleRange=32)
        self.rs_pipe = rs.pipeline()
        self.rs_cfg = rs.config()
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
            frame_mutex.acquire()
            self.frame_data["left"] = left_data
            self.frame_data["right"] = right_data
            self.frame_data["timestamp_ms"] = ts
            frame_mutex.release()

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

    def init_rs_pipe(self):
        self.rs_pipe.start(self.rs_cfg, self._callback)

    def close_rs_pipe(self):
        self.rs_pipe.stop()

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

    def is_valid_frame(self):
        frame_mutex.acquire()
        valid = self.frame_data["timestamp_ms"] is not None
        frame_mutex.release()
        return valid

    def next_frame(self):
        assert self.undistort_rectify != {}, \
            "please init the undistort_rectify before start!"
        frame_mutex.acquire()
        frame_copy = {"left": self.frame_data["left"].copy(),
                      "right": self.frame_data["right"].copy()}
        frame_mutex.release()

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
        return frame[camera][:, self.max_disp:]

    def run(self, height=300, camera="left", save_dir=None, steps=150):
        assert camera in ("left", "right", "both")
        self.init_rs_pipe()
        try:
            i = 0
            self.init_undistort_rectify(height)
            while True:
                # If frames are ready to process
                if self.is_valid_frame():
                    i += 1
                    frame = self.next_frame()

                    if camera in ("left", "right"):
                        color_image = self.get_color_img(frame, camera)
                    elif camera == "both":
                        l_image = self.get_color_img(frame, "left")
                        r_image = self.get_color_img(frame, "right")
                        color_image = np.hstack((l_image, r_image))
                    else:
                        raise("invalid camera name!")
                    # print(color_image.shape)
                    color_image = cv2.cvtColor(color_image, cv2.COLOR_GRAY2RGB)
                    cv2.imshow("color image", color_image)
                    cv2.waitKey(1)
                    if save_dir is not None and os.path.exists(save_dir):
                        if i % steps == 0:
                            save_path = os.path.join(save_dir, str(i)+'.png')
                            cv2.imwrite(save_path, color_image)
        finally:
            self.close_rs_pipe()


if __name__ == "__main__":
    fish_eye = RsFishEye()
    fish_eye.run(camera="both")
