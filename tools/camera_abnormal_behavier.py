# detect jump and sideways. just for showing, not stable.

import sys
sys.path.append(".")

import time
import os
import numpy as np
import cv2
import torch

import pyopenpose as op
import pyrealsense2 as rs

from dataset import Body25Extractor3D, BodyGenerator, OpBody25Dataset

from torchvision.transforms import Compose
from dataset.transforms import BodyExpSmooth, BodyResize
from dataset.transforms import BodyNormalize, BodyCoordTransform, BodyInterpolation

class AbnormalBehavier(object):
    ''' abnormal behavier detection
    '''

    def __init__(self, pose_model_folder, cls_names,
                 width=640, height=480, fps=30,
                 depth_format=rs.format.z16, color_format=rs.format.rgb8):

        self.cls_names = cls_names
        self.extractor = Body25Extractor3D(pose_model_folder, width, height, fps, \
                                           depth_format, color_format)
        self.op_wrapper = self.extractor.op_wrapper
        self.datum = self.extractor.datum
        self.rs_pipe = self.extractor.rs_pipe
        self.rs_cfg = self.extractor.rs_cfg

    def body_keypoints(self, frame):
        '''detect body keypoints from a frame

        Args:
            frame: one frame from realsense, including color and depth frame info
        
        Return:
            pose keypoints array, shape (N, 25, 3)
        
        '''

        return self.extractor.body_keypoints(frame)

    def detect_close(self, point_list, min_d=0.1):
        '''detect close state
        
        Args:
            point_list: [ndarray list], points info of N person, N is the list length
            min_d:[float], the minimum distance of 2 person

        Return:
            result: [ndarray], shape (N, N), close: 1, else 0
        '''
        for i in range(len(point_list)-1):
            person_1 = point_list[i]
            person_2 = point_list[i+1]
            person_1 = person_1[person_1[:,-1]>0.0]
            person_2 = person_2[person_2[:,-1]>0.0]
            center_1 = np.mean(person_1, axis=0)
            center_2 = np.mean(person_2, axis=0)
            d = np.sqrt((center_1[0]-center_2[0])**2 + (center_1[1]-center_2[1])**2 + (center_1[2]-center_2[2])**2)
            print("d = ", d)
            if d< min_d:
                
                return True
        return False

    def detect_jump(self, proposal_seq):
        '''
        Args:
            proposal_seq: shape (N, 25, 4)
        Return:
            - True: jump
            - False: not jump 
        '''
        interest_points = proposal_seq[:,[10, 13], 1] # y, leg
        # print("interest_points = ", interest_points.shape)
        d_y = np.max(interest_points, 0) - np.min(interest_points, 0)
        v_y = np.var(interest_points, 0)
        interest_points = proposal_seq[:,[10, 13], 2] # y, leg
        d_z = np.max(interest_points, 0) - np.min(interest_points, 0)
        

        # print(v_y, d_y)
        # if (max(d1) > 0.15 and max(v1) > 0.5) or (max(d2) > 0.15 and max(v2) > 0.5):
        if max(d_y) > 0.4 and max(v_y) > 0.02 and max(d_z)>0.15:
            return True
        else:
            return False
        
    def detect_sideways(self, proposal_seq):
        '''
        '''
        interest_points1 = proposal_seq[:,[9, 10], :3]
        interest_points2 = proposal_seq[:,[12, 13], :3]
        d_xy = np.mean(interest_points2[:,:,0] - interest_points1[:,:,0])
        z0 = np.mean(interest_points1[0:5,:,-1])
        z1 = np.mean(interest_points1[-5:,:,-1])
        dz = z0 - z1
        # print(dz)
        if np.abs(d_xy) < 0.12 and dz > 0.4:
            return True

        else:
            return False


    def update_points_seq(self, pre_seq, current_points):
        '''
        Args:
            pre_seq: [list of list], [[frame1, frame2, ...], [frame1, frame2, ...], ...]
                each frame is a 25x4 array, [x, y, z, score]
            current_points: [list of array], all keypoints info of current frame [25x4 array, 25x4 array, ...]

        Return:
            updated points sequence, same with pre_seq
        '''
        updated_seq = pre_seq[:]
        if pre_seq == []:
            updated_seq = [[current_point] for current_point in current_points]
            return updated_seq

        for current_point in current_points:
            current_point_array = current_point[:,:3] # [25, 3]
            current_point_center = np.mean(current_point_array, axis=0).reshape([1,3])
            dis_to_pre = []
            for pre_point in pre_seq:
                pre_point_array = np.array(pre_point)[:,:,:3]
                pre_point_center = np.mean(pre_point_array.reshape([-1, 3]), axis=0).reshape([1,3])
                current_dis = np.sqrt(np.sum((current_point_center-pre_point_center)**2))
                dis_to_pre.append(current_dis)
            if min(dis_to_pre) < 5:
                min_idx = np.argmin(dis_to_pre)
                updated_seq[min_idx].append(current_point)
            else:
                updated_seq.append([current_point])

        return updated_seq

            
    def infer_pipeline(self, seq_len=30, op_show=False):
        '''
        '''
        self.rs_pipe.start(self.rs_cfg)
        self.op_wrapper.start()
        align_to = rs.stream.color
        align = rs.align(align_to)

        points_seqs = []
        frame_number = 0
        restart_max = seq_len*2
        restart_num = 0

        while True:
            # 1. get current frame
            frames = self.rs_pipe.wait_for_frames(timeout_ms=1000)
            current_frame_num = frames.get_frame_number()
            if current_frame_num == frame_number:
                continue
            else:
                frame_number = current_frame_num

            # 2. align the color frame and depth frame
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if (not aligned_depth_frame) or (not color_frame):
                continue

            # 3. get keypoints info
            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            keypoints, cv_output = self.body_keypoints(color_image)
            # print(keypoints.shape)
            person_num = 0 if len(keypoints.shape) == 0 else keypoints.shape[0]

            if person_num == 0:
                if op_show is True:
                    cv2.imshow("openpose output", color_image[:,:,::-1])
                    cv2.waitKey(1)
                continue
            else:
                # all keypoints info of current frame [25x4 array, 25x4 array, ...]
                points = []
                for keypoint in keypoints:
                    if keypoint[[10,13],:].any() < 0.1:
                        continue
                    point = self.extractor.keypoints_pixel_to_points(keypoint, aligned_depth_frame)
                    points.append(point)

                # close detect
                if self.detect_close(points, 0.5) is True:
                    print("close detected.")
                    cv2.putText(color_image, "close", (20, 150), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)



            # update keypoints info (series keypoints for detection, eg. 30 frames)
            points_seqs = self.update_points_seq(points_seqs, points)
            # print("points_seqs = ", len(points_seqs))

            seqs_len = [len(points_seq) for points_seq in points_seqs]
            #print(max(seqs_len))
            if len(seqs_len) == 0:
                continue

            if max(seqs_len) >= seq_len:
                # each points in proposal_seqs is a series of N frames, shape (25, 4)
                proposal_seqs = [points_seq for points_seq in points_seqs if len(points_seq) > seq_len*0.7]
                for proposal_seq in proposal_seqs:
                    
                    proposal_seq = np.array(proposal_seq) # shape (N, 25, 4)
                    # print(proposal_seq.shape)
                    
                    # jump detect
                    
                    if self.detect_jump(proposal_seq):
                        # print("jump detected")
                        cv2.putText(color_image, "jump", (20, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)

                    # sideways detect
                    if self.detect_sideways(proposal_seq):
                        # print("sideways detected")
                        cv2.putText(color_image, "sideways", (20, 100), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 2)
                # close detect with all proposals

                # update points_seqs
                # points_seqs = []
                for i, points_seq in enumerate(points_seqs):
                    if len(points_seq) >= seq_len:
                        points_seqs[i] = points_seq[-seq_len+1:]


            if op_show is True:
                cv2.imshow("openpose output", color_image[:,:,::-1])
                cv2.waitKey(1)


if __name__ == "__main__":
    pose_model_folder = "/home/zhoujh/Project/openpose/models"
    cls_names = ['normal', 'jump', 'sideways', 'close']
    abnormal_behavier = AbnormalBehavier(pose_model_folder, cls_names)
    abnormal_behavier.infer_pipeline(seq_len=30, op_show=True)
