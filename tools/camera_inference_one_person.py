import sys
sys.path.append(".")

import time
import os
import numpy as np
import cv2
from collections import Counter
import torch
from torch.nn.functional import softmax
import pyrealsense2 as rs

from mmcv import Config
from dataset import build_dataset, OpenposeExtractor, TrtposeExtractor
from model import build_model, LSTM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CameraInference3D(object):
    '''inference of body keypoints model for video from realsense
    '''

    def __init__(self,
                 cfg_path,
                 checkpoints,
                 seq_len,
                 op_model=None,
                 trt_model=None,
                 pose_json_path=None):
        assert os.path.splitext(checkpoints)[-1] == '.pth'

        cfg = Config.fromfile(cfg_path)
        data_cfg = cfg.dataset
        self.cls_names = data_cfg.test.cls_names
        self.body_names = data_cfg.test.body_names
        self.dataset = build_dataset(data_cfg.test)
        self.idx_list = self.dataset.idx_list
        self.transforms = self.dataset.transforms

        self.gesture_model = build_model(cfg.model).to(device)
        self.gesture_model.load_state_dict(torch.load(checkpoints))
        self.gesture_model.eval()

        self.width = 640
        self.height = 480
        self.pose_w = 224
        self.pose_h = 224
        self.op_wrapper = None

        if op_model is not None:
            self.mode = "openpose"
            self.extractor = OpenposeExtractor(op_model)
            self.pose_wscale = self.width / self.pose_w
            self.pose_hscale = self.height / self.pose_h
            self.dim = 4  # (x, y, z, score)
            self.op_wrapper = self.extractor.op_wrapper
            self.datum = self.extractor.datum
        elif trt_model is not None and pose_json_path is not None:
            self.mode = "trtpose"
            self.extractor = TrtposeExtractor(trt_model,
                                              pose_json_path)
            self.pose_wscale = 1
            self.pose_hscale = 1
            self.dim = 3  # (x, y, z)
        else:
            raise

        self.seq_len = seq_len

        self.rs_pipe = self.extractor.rs_pipe
        self.rs_cfg = self.extractor.rs_cfg

        self.predict = torch.zeros([len(self.cls_names), ])
        self.vote_proposals = []

    def get_location_info(self, keypoints):
        x_idx = self.dim * np.array(self.idx_list)
        y_idx = self.dim * np.array(self.idx_list) + 1
        z_idx = self.dim * np.array(self.idx_list) + 2
        idx_list = sorted(list(x_idx) + list(y_idx) + list(z_idx))
        return keypoints[:, idx_list]

    def body_keypoints(self, color_image):
        '''detect body keypoints from a frame

        Args:
            frame: one frame from realsense, including color and depth frame info

        Return:
            pose keypoints array, shape (N, 25, 3)

        '''
        color_image = cv2.resize(color_image, (self.pose_w, self.pose_h))
        keypoints, cv_output = self.extractor.body_keypoints(color_image)
        if keypoints.shape[0] > 0:
            keypoints[:, :, 0] = keypoints[:, :, 0] * self.pose_wscale
            keypoints[:, :, 1] = keypoints[:, :, 1] * self.pose_hscale
        cv_output = cv2.resize(cv_output, (self.width, self.height))

        return keypoints, cv_output

    def _is_static(self, points_array):
        ''' get static state based on the points array variance
        '''
        points_array = self.get_location_info(points_array)
        var_mean = np.mean(np.var(points_array, axis=0))
        var_max = np.max(np.var(points_array, axis=0))
        # print("var mean = {}, var max = {}".format(var_mean, var_max))
        if var_max < 0.001 and var_mean < 0.0001:
            # print("static: var_max = {}, var_mean = {}".format(var_max, var_mean))
            return True
        else:
            return False

    def _post_process_wave(self, points_array):
        '''futher assertion for wave

        Args:
            points_array: x,y,z info of 'Neck', 'RShoulder', 'RElbow', 'RWrist', (T, 12)

        Return:
            True / False: wave or not
        '''
        print("in post procss: wave")
        if self.mode == "openpose":
            rwrist_x = points_array[:, 9]
            rwrist_y = points_array[:, 10]
            rwrist_z = points_array[:, 11]
            relbow_y = points_array[:, 7]
            rshoulder_y = points_array[:, 4]
        else:
            rwrist_x = points_array[:, 6]
            rwrist_y = points_array[:, 7]
            rwrist_z = points_array[:, 8]
            relbow_y = points_array[:, 4]
            rshoulder_y = points_array[:, 1]
        rwrist_x = rwrist_x[rwrist_x != 0]
        rwrist_y = rwrist_y[rwrist_y != 0]
        rwrist_z = rwrist_z[rwrist_z != 0]
        relbow_y = relbow_y[relbow_y != 0]
        rshoulder_y = rshoulder_y[rshoulder_y != 0]

        if rwrist_x.shape[0] * rwrist_y.shape[0] * rwrist_z.shape[0] * \
           relbow_y.shape[0] * rshoulder_y.shape[0] == 0:
            return False

        if np.mean(rwrist_y) >= np.mean(relbow_y):
            print("wave: too high relbow_y")
            return False

        if np.max(rwrist_x) - np.min(rwrist_x) < 0.05:
            print("wave: too small rwrist_x")
            return False

        if np.max(rwrist_z) - np.min(rwrist_z) > 0.15:
            print("wave: too large rwrist_z")
            return False

        return True

    def _post_process_come(self, points_array):
        '''futher assertion for come

        Args:
            points_array: x,y,z info of 'Neck', 'RShoulder', 'RElbow', 'RWrist', (T, 12)

        Return:
            True / False: come or not
        '''
        print("in post procss: come")
        if self.mode == "openpose":
            rwrist_x = points_array[:, 9]
            rwrist_y = points_array[:, 10]
            rwrist_z = points_array[:, 11]
            relbow_y = points_array[:, 7]
            rshoulder_y = points_array[:, 4]
        else:
            rwrist_x = points_array[:, 6]
            rwrist_y = points_array[:, 7]
            rwrist_z = points_array[:, 8]
            relbow_y = points_array[:, 4]
            rshoulder_y = points_array[:, 1]

        rwrist_x = rwrist_x[rwrist_x != 0]
        rwrist_y = rwrist_y[rwrist_y != 0]
        rwrist_z = rwrist_z[rwrist_z != 0]
        relbow_y = relbow_y[relbow_y != 0]
        rshoulder_y = rshoulder_y[rshoulder_y != 0]

        if rwrist_x.shape[0] * rwrist_y.shape[0] * rwrist_z.shape[0] * \
           relbow_y.shape[0] * rshoulder_y.shape[0] == 0:
            return False
        if np.mean(rwrist_y) - np.mean(relbow_y) > 0.05:
            print("come: too low relbow")
            return False
        if np.mean(rshoulder_y) >= np.mean(relbow_y):
            print("come: too high relbow")
            return False
        if np.max(rwrist_x) - np.min(rwrist_x) > 0.18:
            print("come: too large rwrist_x")
            return False
        if np.max(rwrist_z) - np.min(rwrist_z) < 0.1:
            print("come: too small rwrist_z")
            return False

        return True

    def _post_process_hello(self, points_array):
        '''futher assertion for hello

        Args:
            points_array: x,y,z info of 'Neck', 'RShoulder', 'RElbow', 'RWrist', (T, 12)

        Return:
            True / False: come or not
        '''
        print("in post procss: hello")
        if self.mode == "openpose":
            rwrist_y = points_array[:, 10]
            neck_y = points_array[:, 1]
        else:
            rwrist_y = points_array[:, 7]
            neck_y = points_array[:, 10]

        rwrist_y = rwrist_y[rwrist_y != 0]
        neck_y = neck_y[neck_y != 0]
        num1 = rwrist_y.shape[0] // 10
        num2 = neck_y.shape[0] // 10

        if rwrist_y.shape[0] * neck_y.shape[0] == 0:
            return False

        if num1 < 1 or num2 < 1:
            return False

        if rwrist_y.shape[0] < points_array.shape[0] // 3:
            return False

        if np.mean(rwrist_y[0:num1] - np.mean(rwrist_y[-num1:])) < 0.3:
            print("hello: too low for hello")
            return False

        if np.mean(rwrist_y[-num1:]) > np.mean(neck_y[-num2:]):
            print("hello: lower than neck")
            return False

        return True             

    def _post_process(self, points_array, label):
        if label == "wave":
            return self._post_process_wave(points_array)
        elif label == "come":
            return self._post_process_come(points_array)
        elif label == "hello":
            return self._post_process_hello(points_array)
        else:
            return True

    def _predict(self, points_array):
        points_tensor = self.transforms(points_array)
        points_tensor = torch.Tensor(points_tensor).unsqueeze(0).cuda()
        with torch.no_grad():
            predict = self.gesture_model(points_tensor)[0].cpu()
            predict = softmax(predict)

        if max(predict) > 0.5:
            predict = self._smooth_predict(predict)
            predict_label = self.cls_names[np.argmax(predict)]
            flag = self._post_process(points_array, predict_label)
            predict_label = predict_label if flag is True else "others"
            predict_label = self._vote(predict_label)
        else:
            print("ignore low score results...")
            predict_label = "others"
        return predict_label

    def _smooth_predict(self, predict, alpha=0.5):
        predict = softmax((1-alpha) * self.predict + alpha * predict)
        self.predict = predict
        return predict

    def _vote(self, predict_label, num=5):
        self.vote_proposals.append(predict_label)
        if len(self.vote_proposals) < num:
            return predict_label
        else:
            self.vote_proposals.append(predict_label)
            self.vote_proposals = self.vote_proposals[-num:]
            return Counter(self.vote_proposals).most_common(1)[0][0]

    def _draw_result(self, image, predict_label, t):
        show_image = np.array(image[:,:,::-1], dtype=np.uint8)
        if predict_label in self.cls_names[:-1]:
            cv2.putText(show_image, predict_label, (20, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        speed = str(int(1 / t)) + ' FPS'
        cv2.putText(show_image, speed, (show_image.shape[1]-100, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2)
        return show_image

    def _get_angle(self, points_array):
        raise NotImplementedError

    def infer_pipeline(self, show=False):
        '''inference pipeline for frames stream from realsense
        '''

        self.rs_pipe.start(self.rs_cfg)
        if self.op_wrapper is not None:
            self.op_wrapper.start()

        point_list = []
        frame_number = 0
        while True:
            predict_label = 'others'
            # get a frame of color image and depth image
            frames = self.rs_pipe.wait_for_frames(timeout_ms=5000)
            current_frame_num = frames.get_frame_number()
            if current_frame_num == frame_number:
                continue
            else:
                frame_number = current_frame_num

            depth_frame, color_frame = self.extractor._get_aligned_frame(frames)

            if (not depth_frame) or (not color_frame):
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # get 3-D points info with openpose
            time_s = time.time()
            keypoints, cv_output = self.body_keypoints(color_image)
            # op_time = time.time() - time_s
            # print("openpose speed = {} FPS".format(int(1/op_time)))

            if keypoints.shape[0] > 0:
                keypoint = self.extractor.get_best_keypoint(keypoints)
                point = self.extractor.keypoint_to_point(keypoint, depth_frame)
                point_list.append(point.flatten())

            # lstm inference
            if len(point_list) >= self.seq_len:
                point_array = np.array(point_list, dtype='float32')
                if self._is_static(point_array):
                    print("static state...")
                    predict_label = "others"
                else:
                    lstm_time = time.time()
                    point_array = self.get_location_info(point_array)
                    predict_label = self._predict(point_array)
                    lstm_time = time.time() - lstm_time
                    # print("lstm speed = {} FPS".format(int(1/(lstm_time))))
                    print("label = {}".format(predict_label))
                point_list = point_list[-self.seq_len+1:]
            time_e = time.time()
            # result visualization
            if show is True:
                show_img = self._draw_result(cv_output, predict_label, time_e-time_s)
                cv2.imshow("gesture output", show_img)
                cv2.waitKey(1)
        if self.op_wrapper is not None:
            self.op_wrapper.stop()


if __name__ == "__main__":
    # openpose model
    # pose_model_folder = "/home/zhoujh/Project/openpose/models"
    # checkpoints = "/home/zhoujh/Project/gesture_workdir/op_body25/lstm_ce/checkpoints/model_5000.pth"
    # cfg_path = "/home/zhoujh/Project/gesture_workdir/op_body25/lstm_ce/lstm_op-body25.py"
    # seq_len = 30
    # camera_infer = CameraInference3D(cfg_path=cfg_path,
    #                                  checkpoints=checkpoints,
    #                                  seq_len=seq_len,
    #                                  op_model=pose_model_folder,
    #                                  )

    # trt-pose model
    trt_model = "/home/zhoujh/Project/gesture_recognition/checkpoints/resnet18_baseline_att_224x224_A_epoch_249_trt.pth"
    checkpoints = "/home/zhoujh/Project/gesture_workdir/trt_body18/lstm_ce/checkpoints/model_5000.pth"
    pose_json_path = "/home/zhoujh/Project/gesture_recognition/test_code/human_pose.json"
    cfg_path = "/home/zhoujh/Project/gesture_workdir/trt_body18/lstm_ce/lstm_trt-body18.py"
    seq_len = 30
    camera_infer = CameraInference3D(cfg_path=cfg_path,
                                     checkpoints=checkpoints,
                                     seq_len=seq_len,
                                     trt_model=trt_model,
                                     pose_json_path=pose_json_path
                                     )

    camera_infer.infer_pipeline(show=True)
