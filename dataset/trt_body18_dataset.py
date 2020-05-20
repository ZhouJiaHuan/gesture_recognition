# Description: custom dataset for gesture recognition
# Author: ZhouJH
# Data: 2020/4/8

import numpy as np
from .body_dataset import BodyDataset
from .registry import DATASET


@DATASET.register_module
class TrtBody18Dataset(BodyDataset):
    '''

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

    BODY_NAMES = tuple(["nose", "left_eye", "right_eye", "left_ear",
                        "right_ear", "left_shoulder", "right_shoulder",
                        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                        "left_hip", "right_hip", "left_knee", "right_knee",
                        "left_ankle", "right_ankle", "neck"])

    def __init__(self, *args, **kwargs):
        super(TrtBody18Dataset, self).__init__(*args, **kwargs)

    def _get_location_info(self, keypoints):
        x_idx = 3 * np.array(self.idx_list)
        y_idx = 3 * np.array(self.idx_list) + 1
        z_idx = 3 * np.array(self.idx_list) + 2
        idx_list = sorted(list(x_idx) + list(y_idx) + list(z_idx))
        return keypoints[:, idx_list]
