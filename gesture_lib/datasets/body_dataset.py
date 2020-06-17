# Description: custom dataset for gesture recognition
# Author: ZhouJH
# Data: 2020/4/8

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from abc import abstractmethod
from .registry import DATASET
from .transforms import Compose


@DATASET.register_module
class BodyDataset(Dataset):
    '''
    data tree:

    Data_Folder
        - gesture1
            - gesture1_1.txt
            - gesture1_2.txt
            - ...
        - gesture2
            - gesture2_1.txt
            - gesture2_2.txt
            - ...
        - ...

    each txt save a series of keypoints info (transformed) shaped (T, D), 
    where T is the series length and D is the feature dimention.
    '''

    BODY_NAMES = ['Nose', 'Neck', 'RShoulder', 'RElbow', 'RWrist',
                  'LShoulder', 'LElbow', 'LWrist', 'MidHip', 'RHip',
                  'RKnee', 'RAnkle', 'LHip', 'LKnee', 'LAnkle',
                  'REye', 'LEye', 'REar', 'LEar', 'LBigToe',
                  'LSmallToe', 'LHeel', 'RBigToe', 'RSmallToe', 'RHeel',
                  ]

    def __init__(self,
                 data_folder,
                 cls_names,
                 body_names=None,
                 transforms=None):
        assert len(self.BODY_NAMES) > 0
        super(BodyDataset, self).__init__()
        self.cls_names = tuple(cls_names)
        self.txt_list = []
        self.points_num = len(self.BODY_NAMES)
        if body_names is None:
            self.idx_list = list(range(self.points_num))
        else:
            self.idx_list = [self.BODY_NAMES.index(body_name)
                             for body_name in body_names
                             if body_name in self.BODY_NAMES]
        for cls_name in self.cls_names:
            cls_folder = os.path.join(data_folder, cls_name)
            if os.path.isdir(cls_folder):
                self.txt_list.extend(glob.glob(os.path.join(cls_folder, '*.txt')))

        self.transforms = Compose(transforms)

    @abstractmethod
    def _get_location_info(self, keypoints):
        '''extract x, y, z info from keypoints array with specified body idx list
        '''
        raise NotImplementedError

    def __len__(self):
        return len(self.txt_list)

    def __getitem__(self, idx):
        txt_path = self.txt_list[idx]
        cls_name = os.path.basename(txt_path).split('_')[0]
        label = torch.tensor(self.cls_names.index(cls_name))

        keypoints = self._get_location_info(np.loadtxt(txt_path))
        if self.transforms is not None:
            keypoints = self.transforms(keypoints)
        keypoints = torch.tensor(keypoints, dtype=torch.float32)

        sample = {'keypoints': keypoints, 'label': label}
        return sample
