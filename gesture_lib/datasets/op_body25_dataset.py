# Description: custom dataset for gesture recognition
# Author: ZhouJH
# Data: 2020/4/8

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from gesture_lib.registry import DATASET
from .body_dataset import BodyDataset
from .transforms import *


@DATASET.register_module(name="OpBody25Dataset")
class OpBody25Dataset(BodyDataset):
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

    def __init__(self, *args, **kwargs):
        super(OpBody25Dataset, self).__init__(*args, **kwargs)

    def _get_location_info(self, keypoints):
        x_idx = 4 * np.array(self.idx_list)
        y_idx = 4 * np.array(self.idx_list) + 1
        z_idx = 4 * np.array(self.idx_list) + 2
        idx_list = sorted(list(x_idx) + list(y_idx) + list(z_idx))
        return keypoints[:, idx_list]
