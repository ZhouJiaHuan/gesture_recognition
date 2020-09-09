import numpy as np
from gesture_lib.ops.keypoint import get_location, get_keypoint_box


class MemoryManager(object):
    ''' manage the person infomation in tracking process

    the infomation includes the skeletons and features, which
    are used for person matching with the feature extracted
    in tracking process.

    concretely, the `MemoryManager` manages 2 containers, namely
    `memory` and `cache`, which are recorded with dict and
    list respectivaly. When meeting a new person, the skeletons and
    features will be extracted and stored in `cache` temporary.
    If the pop condition satisfied before cleaning, the skeletons and
    features will be put into `memory`. Otherwise, the temporary info
    will be cleaned when cleaning condition satisfied.

    each person in `memory` is recorded with a dict, which contains:

        - keypoint: keypoint array. eg, for trtpose, shape 18x2
        
        - point: point array. eg, for trtpose, shape 18x2
        
        - keypoint_box: minimum bounding rectangle, (x1, y1, x2, y2)
        
        - point_center: 3-D location (x, y, z)
        
        - keypoint_feature: feature extracted from color image with
            keypoint info. The type is decided by `extract_feature`
    '''
    def __init__(self, matcher, capacity=5, max_id=100, clean_times=20,
                 pop_num=4, sim_thr1=0.2, sim_thr2=0.8):
        self.matcher = matcher
        self.capacity = capacity
        self.max_id = max_id
        self.clean_times = clean_times
        self.pop_num = pop_num
        self.memory = {}
        self.cache = []
        self.memory_count = 0
        self.memory_pop_id = 1
        self.cache_count = 0
        self.sim_thr1 = sim_thr1  # for memory
        self.sim_thr2 = sim_thr2  # for memory cache

    def memory_ids(self):
        return list(self.memory.keys())

    def _extract_feature(self, img_bgr, keypoint):
        '''extract feature from color image with specified keypoint info.
        '''
        return self.matcher.extract_feature(img_bgr, keypoint)

    def prepare_input(self, color_image, skeleton):
        input_info = {}
        keypoint, point = skeleton
        keypoint_feature = self._extract_feature(color_image, keypoint)
        point_center = get_location(point)
        input_info['keypoint'] = keypoint
        input_info['point'] = point
        input_info['keypoint_box'] = get_keypoint_box(keypoint)
        input_info['point_center'] = point_center
        input_info['keypoint_feature'] = keypoint_feature
        input_info['visible'] = True
        return input_info

    def _person_sim(self, memory_info, input_info):
        ''' compute the feature similarity based on memory and input_info

        the memory_info and input_info are dicts, which at least contain
        following infomation:
        - keypoint: keypoint array, for trtpose, shape 18x2
        - point: point array, for trtpose, shape 18x2
        - keypoint_box: minimum bounding rectangle, (x1, y1, x2, y2)
        - point_center: 3-D location (x, y, z)
        - keypoint_feature: feature vector extracted from color image with
            keypoint info. The type is decided by `_extract_feature`
        '''
        return self.matcher.match(memory_info, input_info)

    def find_best_match(self, input_info):
        best_person_id = '0'
        best_sim = 0

        for person_id, memory_info in self.memory.items():
            temp_sim = self._person_sim(memory_info, input_info)
            if temp_sim > best_sim:
                best_sim = temp_sim
                best_person_id = person_id
        return best_person_id, best_sim

    def update_person_memory(self, person_id, input_info):
        if person_id in self.memory.keys():
            self.memory[person_id]['keypoint'] = input_info['keypoint']
            self.memory[person_id]['point'] = input_info['point']
            self.memory[person_id]['keypoint_box'] = input_info['keypoint_box']
            self.memory[person_id]['point_center'] = input_info['point_center']
            self.memory[person_id]['visible'] = input_info['visible']
        else:
            return

    def update_cache(self, input_info):
        self.cache_count += 1
        if self.cache_count > self.clean_times:
            self.cache_count = 0
            self.cache = []

        for idx, temp_info in enumerate(self.cache):
            temp_sim = self._person_sim(temp_info[0], input_info)
            if temp_sim > self.sim_thr2:
                self.cache[idx][1] += 1
                self.cache[idx][0] = input_info
                count = self.cache[idx][1]
                if count >= self.pop_num:
                    self.cache.pop(idx)
                    return True

        self.cache.append([input_info, 1])
        return False

    def reset_memory_state(self):
        for person_id in self.memory.keys():
            self.memory[person_id]['visible'] = False

    def reset_memory_info(self):
        for person_id in self.memory.keys():
            visible = self.memory[person_id]['visible']
            if visible is True:
                continue
            self.memory[person_id]['keypoint'] *= 0
            self.memory[person_id]['point'] *= 0
            self.memory[person_id]['keypoint_box'] = np.zeros(4)
            self.memory[person_id]['point_center'] = np.zeros(3)

    def pop_front_or_not(self):
        if len(self.memory) >= self.capacity:
            self.memory.pop(str(self.memory_pop_id), 0)
            self.memory_pop_id = self.memory_pop_id % self.max_id + 1
            return True
        else:
            return False

    def push_back(self, input_info):
        person_id = str(self.memory_count % self.max_id + 1)
        self.memory[person_id] = input_info
        self.memory_count += 1
        return person_id
