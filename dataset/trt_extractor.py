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
from .body_extractor import BodyExtractor

device = torch.device('cuda')


class TrtposeExtractor(BodyExtractor):
    '''
    '''

    def __init__(self,
                 trt_model,
                 pose_json_path,
                 *args,
                 **kwargs):
        super(TrtposeExtractor, self).__init__(*args, **kwargs)

        with open(pose_json_path, 'r') as f:
            human_pose = json.load(f)
        self.keypoints = human_pose['keypoints']
        self.keypoints_num = len(self.keypoints)
        self.topology = trt_pose.coco.coco_category_to_topology(human_pose)

        self.model_trt = TRTModule()
        self.model_trt.load_state_dict(torch.load(trt_model))
        self.model_trt.to(device)
        self.parse_objects, self.draw_objects = self._config_trt()

    def _config_trt(self):
        parse_objects = ParseObjects(self.topology)
        draw_objects = DrawObjects(self.topology)
        return parse_objects, draw_objects

    def _preprocess(self, image):
        mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
        std = torch.Tensor([0.229, 0.224, 0.225]).cuda()
        image = cv2.resize(image, (224, 224))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = PIL.Image.fromarray(image)
        image = transforms.functional.to_tensor(image).to(device)
        image.sub_(mean[:, None, None]).div_(std[:, None, None])
        return image[None, ...]

    def body_keypoints(self, img_process):
        '''detect body keypoints

        Args:
            img_process: [ndarray] image array

        Return:
            pose keypoints array, shape (N, point_num, 2)
            cvOutputData: detect result
        '''

        if img_process is None:
            return np.zeros([0, self.keypoints_num, 2]), img_process

        cvOutputData = img_process.copy()
        data = self._preprocess(cvOutputData)
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
                    x = float(peak[1]) * self.width
                    y = float(peak[0]) * self.height
                    keypoints_array[i, j, :] = [x, y]

        self.draw_objects(cvOutputData, counts, objects, peaks)

        return keypoints_array, cvOutputData

    def get_best_keypoint(self, keypoint):
        '''get the body keypoints with maximum score in one frame

        for keypoints extraction

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

        if not (zeros_sum.all() == zeros_sum[0]):
            idx = np.argmax(zeros_sum)
            return keypoint[idx, :, :]
        else:
            x_to_center = np.abs(np.mean(keypoint[:, :, 0], axis=-1) - self.width / 2)
            idx = np.argmin(x_to_center)
            return keypoint[idx, :, :]

    def keypoint_to_point(self, keypoint, depth_frame):
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
            point[i, :3] = self.pixel_to_point(depth_frame, keypoint[i, ])

        return point
