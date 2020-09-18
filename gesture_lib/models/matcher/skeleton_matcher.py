import numpy as np
from .base_matcher import BaseMatcher
from gesture_lib.registry import MATCHERS


@MATCHERS.register_module(name="SkeletonMatcher")
class SkeletonMatcher(BaseMatcher):
    ''' matcher with 3D skeletons
    TODO:
        learning a similarity function from skeletons (similar with Siamese),
        not only the skeletons distance
    '''

    def __init__(self, **kwargs):
        super(SkeletonMatcher, self).__init__(**kwargs)

    def extract_feature(self, img_bgr, skeleton):
        if self.mode == 'openpose':
            point = skeleton[1][[1, 2, 5, 9, 12], :3]
        elif self.mode == 'trtpose':
            point = skeleton[1][[17, 6, 5, 12, 11], :]

        if np.min(point[:, -1]) == 0:  # zero value
            return np.zeros([0, 10])

        feature = []
        for i in range(point.shape[0]-1):
            temp = point[i, ]
            res = point[i+1:, ]
            dis = np.sqrt(np.sum((temp-res)**2, axis=1))
            feature.extend(dis)
        feature = np.reshape(np.array(feature), (1, -1))
        L = np.sqrt(np.sum(feature**2))
        return feature / L

    def feature_similarity(self, feature1, feature2):
        sim = 0
        print(feature2.shape)
        if feature1.shape[0] == 0 or feature2.shape[0] == 0:
            return sim
        sim = 1 - np.linalg.norm(feature1 - feature2, ord=2)
        return sim
