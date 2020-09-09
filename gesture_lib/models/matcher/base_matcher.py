from gesture_lib.ops.box import box_iou
from abc import abstractmethod
from gesture_lib.registry import MATCHERS


@MATCHERS.register_module(name="BaseMatcher")
class BaseMatcher(object):

    def __init__(self, alpha=0.5):
        super(BaseMatcher, self).__init__()
        self.alpha = alpha

    @abstractmethod
    def extract_feature(self, img_bgr, keypoint):
        '''extract feature from BGR image with specified keypoint.

        Args:
            img_bgr: [3-d array], source BGR image/frame
            keypoint: [2-d array], keypoint info, shape (25, 3) for
                openpose body-25 and shape (18, 2) for trtpose body-18
        
        Return:
            feature: feature extracted for `feature_similarity`
            
        '''
        raise NotImplementedError

    @abstractmethod
    def feature_similarity(self, feature1, feature2):
        ''' compute the simlarity of 2 features

        feature1 and feature2 are extracted by `extract_feature`
        '''
        raise NotImplementedError

    def match(self, person1, person2):
        '''compute the similarity of 2 persons

        each person is recorded with a dict, which contains:

        - keypoint: keypoint array. eg, for trtpose, shape 18x2
        
        - point: point array. eg, for trtpose, shape 18x2
        
        - keypoint_box: minimum bounding rectangle, (x1, y1, x2, y2)
        
        - point_center: 3-D location (x, y, z)
        
        - keypoint_feature: feature extracted from color image with
            keypoint info. The type is decided by `extract_feature`

        - visible: if the person is found in current frame
        '''

        sim = 0
        box1 = person1['keypoint_box']
        box2 = person2['keypoint_box']
        if box1[0] > box1[2] or box1[1] > box1[3] or \
           box2[0] > box2[2] or box2[1] > box2[3]:
            # invalid box
            return sim
        
        diou = box_iou(box1, box2, mode='diou')

        feature1 = person1['keypoint_feature']
        feature2 = person2['keypoint_feature']
        feature_sim = self.feature_similarity(feature1, feature2)
        sim = self.alpha * feature_sim + (1-self.alpha) * diou

        return sim

