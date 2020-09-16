import cv2
import numpy as np
import json
import PIL
import torch
from torchvision import transforms

from torch2trt import TRTModule
import trt_pose.coco
import trt_pose.models
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects

device = torch.device('cuda')


class TrtPose(object):
    ''' ops of trt-pose for body-18 pose estimation
    '''
    def __init__(self, trt_model, pose_json_path):
        with open(pose_json_path, 'r') as f:
            human_pose = json.load(f)
        self.keypoints = human_pose['keypoints']
        self.keypoints_num = len(self.keypoints)

        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(trt_model))
        self.model_trt.to(device)

        topology = trt_pose.coco.coco_category_to_topology(human_pose)
        self.parse_objects = ParseObjects(topology)
        self.draw_objects = DrawObjects(topology)

    def _preprocess(self, img, resize):
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        img = cv2.resize(img, resize)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
        img = transforms.functional.to_tensor(img).to(device)
        img.sub_(mean[:, None, None]).div_(std[:, None, None])
        return img[None, ...]

    def body_keypoints(self, img, resize=(224, 224)):
        '''detect body keypoints

        Args:
            img: [ndarray] image array

        Return:
            pose keypoints array, shape (N, point_num, 2)
            cvOutputData: detect result
        '''
        h, w = img.shape[:2]
        if img is None:
            return np.zeros([0, self.keypoints_num, 2]), img

        cvOutputData = img.copy()
        data = self._preprocess(cvOutputData, resize)
        cmap, paf = self.model_trt(data)
        cmap = cmap.detach().cpu()
        paf = paf.detach().cpu()
        counts, objects, peaks = self.parse_objects(cmap, paf)
        person_num = int(counts[0])

        keypoints_array = np.zeros((person_num, self.keypoints_num, 2))
        for i in range(person_num):
            obj = objects[0][i]
            for j in range(self.keypoints_num):
                k = int(obj[j])
                if k >= 0:
                    peak = peaks[0][j][k]
                    x = float(peak[1]) * w
                    y = float(peak[0]) * h
                    keypoints_array[i, j, :] = [x, y]

        self.draw_objects(cvOutputData, counts, objects, peaks)

        return keypoints_array, cvOutputData

    def get_best_keypoint(self, keypoint):
        '''get the body keypoints with maximum score in one frame

        for keypoints extraction

        TODO:
            find better strategy

        Args:
            pose_keypoints: [ndarray], shape (N, 18, 2)

        Return:
            pose_keypoint: [ndarray], shape (18, 2)
        '''

        person_num = keypoint.shape[0]
        assert keypoint.shape[0] > 0

        if person_num == 1:
            return keypoint[0, :, :]

        no_zeros = keypoint.copy().reshape([person_num, -1])
        no_zeros[no_zeros > 0] = 1
        zeros_sum = np.sum(no_zeros, axis=-1)

        idx = np.argmax(zeros_sum)
        return keypoint[idx, :, :]
