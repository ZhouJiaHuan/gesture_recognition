import time
import numpy as np
import cv2
from numpy import sqrt as np_sqrt
from numpy import sum as np_sum
from gesture_lib.ops.math import cross_point_in, computeP3withD
from gesture_lib.ops.optimize import local_search


class SkeletonGenerator(object):
    ''' 3D skeletons generator with Realsense D435 and T265
    The global 2D skeletons are estimated from T265 fisheye camera
    and the local 3D skeletons are estimated from D435 RGBD camera.
    the fisheye frame can be matched to rgbd frame with a perspective
    matrix M

    As a result, the global 3D skeletons are inferenced with global
    2D skeletons and 3D skeletons
    '''

    def __init__(self, extractor):
        self.M = np.loadtxt("gesture_lib/data/perspMatrix.txt")
        self.extractor = extractor
        self.width = self.extractor.width
        self.height = self.extractor.height
        self.size = (self.width, self.height)
        self.corners = [[(0, 0), (0, self.height)],
                        [(0, self.height), (self.width, self.height)],
                        [(self.width, self.height), (self.width, 0)],
                        [(self.width, 0), (0, 0)]]

    def trans_kp_fish2rgbd(self, fish_kp, fish_img, rgbd_img, k_size=(5, 5)):
        point_num = fish_kp.shape[0]
        rgbd_kp = np.zeros_like(fish_kp)
        h, w = fish_img.shape
        for i in range(point_num):
            if fish_kp[i, 0] < 1e-5 and fish_kp[i, 1] < 1e-5:
                continue
            temp_kp = np.reshape(fish_kp[i, :], (1, 1, -1))
            rgbd_kp[i] = cv2.perspectiveTransform(temp_kp, self.M)
            if fish_kp[i, 0] > k_size[0]//2 and fish_kp[i, 0] < w-k_size[0]//2 \
               and fish_kp[i, 1] > k_size[1]//2 and fish_kp[i, 1] < h-k_size[1]//2:
                x1 = int(fish_kp[i, 0]) - k_size[0]//2
                x2 = int(fish_kp[i, 0]) + k_size[0]//2
                y1 = int(fish_kp[i, 1]) - k_size[1]//2
                y2 = int(fish_kp[i, 1]) + k_size[1]//2
                # kernel = fish_img[y1:y2+1, x1:x2+1]
                # t1 = time.time()
                # rgbd_kp[i] = local_search(kernel, rgbd_img, rgbd_kp[i])
                # t2 = time.time() - t1
                # print("time: t2", t2, ' ----------------------')
        return rgbd_kp

    def _local_search_bak(self, kernel, rgbd_img, kp, search_size=(25, 25)):
        ''' keypoint local search
        Args:
            kernel: a small kernel cropped from fish image
            rgbd_img: gray image from rgbd camera
            kp: current keypoint on rgbd image mapped from fisheye image
            search_size: search area size on rgbd image

        Return:
            result_kp: optimized keypoint on rgbd image
        '''
        def dist(region, kernel):
            print(region.shape, kernel.shape)
            assert region.shape == kernel.shape
            return np.sqrt(np.sum((region-kernel)**2))
 
        k_h, k_w = kernel.shape[:2]
        s_h, s_w = search_size
        img_h, img_w = rgbd_img.shape[:2]
        # case 0: kernel size is larger than search size
        if k_h >= s_h or k_w >= s_w:
            return kp
        # case 1: source keypoint is out of the rgbd image
        if kp[0] < 0 or kp[0] > img_w-1 or kp[1] < 0 or kp[1] > img_h-1:
            return kp
        x1 = max(k_w // 2, int(kp[0] - s_w // 2 + k_w // 2))
        y1 = max(k_h // 2, int(kp[1] - s_h // 2 + k_h // 2))
        x2 = min(img_w-k_w // 2, int(kp[0] + s_w // 2 - k_w // 2))
        y2 = min(img_h-k_h // 2, int(kp[1] + s_h // 2 - k_h // 2))

        # case 2: invalid search region
        if x1 >= x2 or y1 >= y2:
            return kp

        # case 3:
        result_kp = np.asarray(kp, np.int32)
        min_d = np.Inf
        for y in range(y1, y2):
            for x in range(x1, x2):
                temp = rgbd_img[y-k_h//2:y+k_h//2+1, x-k_w//2:x+k_w//2+1]
                temp_d = dist(temp, kernel)
                if temp_d < min_d:
                    min_d = temp_d
                    result_kp = [x, y]
        return result_kp

    def _local_search(self, kernel, rgbd_img, kp, search_size=(25, 25)):
        ''' keypoint local search
        Args:
            kernel: a small kernel cropped from fish image
            rgbd_img: gray image from rgbd camera
            kp: current keypoint on rgbd image mapped from fisheye image
            search_size: search area size on rgbd image

        Return:
            result_kp: optimized keypoint on rgbd image
        '''
        k_h, k_w = kernel.shape[:2]
        s_h, s_w = search_size
        img_h, img_w = rgbd_img.shape[:2]

        def dist(x, y):
            region = rgbd_img[y-k_h//2:y+k_h//2+1, x-k_w//2:x+k_w//2+1]
            print(region.shape, kernel.shape)
            return np_sqrt(np_sum((region-kernel)**2))

        # case 0: kernel size is larger than search size
        if k_h >= s_h or k_w >= s_w:
            return kp
        # case 1: source keypoint is out of the rgbd image
        if kp[0] < 0 or kp[0] > img_w-1 or kp[1] < 0 or kp[1] > img_h-1:
            return kp
        x1 = max(k_w // 2, int(kp[0] - s_w // 2 + k_w // 2))
        y1 = max(k_h // 2, int(kp[1] - s_h // 2 + k_h // 2))
        x2 = min(img_w-k_w // 2, int(kp[0] + s_w // 2 - k_w // 2))
        y2 = min(img_h-k_h // 2, int(kp[1] + s_h // 2 - k_h // 2))

        # case 2: invalid search region
        if x1 >= x2 or y1 >= y2:
            return kp

        # case 3:
        xx = list(range(x1, x2)) * (y2-y1+1)
        yy = list(range(y1, y2)) * (x2-x1+1)
        dists = list(map(lambda x, y: dist(x, y), xx, yy))
        idx = dists.index(min(dists))
        result_kp = [xx[idx], yy[idx]]
        return result_kp

    def keypoint_to_point(self, rgbd_frame, keypoint):
        depth_frame = rgbd_frame['depth']
        point = self.extractor.keypoint_to_point(keypoint, depth_frame)
        return point

    def generate(self, rgbd_frame, src_point, fish_kp, rgbd_kp):
        '''
        Args:
            rgbd_frame: frame data from rgbd camera, include color frame
                and depth frame
            src_point: source 3D skeletons
            fish_kp: 2D keypoints from fisheye camera
            rgbd_kp: 2D keypoints in rgbd camera mapped from fish_kp, some
                of the keypoints may be out of the rgbd frame

        Return:
            dst_point: generated 3D skeletons
        '''
        dst_point = src_point.copy()
        args = {'src_point': dst_point, 'rgbd_frame': rgbd_frame,
                'fish_kp': fish_kp, 'rgbd_kp': rgbd_kp}
        if dst_point[6, -1] == 0:
            dst_point, _ = self.gen_rshoulder(**args)
            args['src_point'] = dst_point
        if dst_point[5, -1] == 0:
            dst_point, _ = self.gen_lshoulder(**args)
            args['src_point'] = dst_point
        if dst_point[17, -1] == 0:
            dst_point, _ = self.gen_nect(**args)
            args['src_point'] = dst_point
        if dst_point[8, -1] == 0:
            dst_point, _ = self.gen_relbow(**args)
            args['src_point'] = dst_point
        if dst_point[7, -1] == 0:
            dst_point, _ = self.gen_lelbow(**args)
            args['src_point'] = dst_point
        if dst_point[10, -1] == 0:
            dst_point, _ = self.gen_rwrist(**args)
            args['src_point'] = dst_point
        if dst_point[9, -1] == 0:
            dst_point, _ = self.gen_lwrist(**args)
            args['src_point'] = dst_point
        if dst_point[14, -1] == 0:
            dst_point, _ = self.gen_rknee(**args)
            args['src_point'] = dst_point
        if dst_point[13, -1] == 0:
            dst_point, _ = self.gen_lknee(**args)
            args['src_point'] = dst_point
        if dst_point[16, -1] == 0:
            dst_point, _ = self.gen_rankle(**args)
            args['src_point'] = dst_point
        if dst_point[15, -1] == 0:
            dst_point, _ = self.gen_lankle(**args)
            args['src_point'] = dst_point
        return dst_point

    def gen_nect(self, src_point, rgbd_frame=None, fish_kp=None, rgbd_kp=None):
        if src_point[17, -1] > 0:
            return src_point, True
        dst_point = src_point.copy()
        if src_point[6, -1] > 0 and src_point[5, -1] > 0:
            dst_point[17, ] = (dst_point[5, ] + dst_point[6, ]) / 2
            return dst_point, True
        return dst_point, False

    def gen_rshoulder(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        if src_point[6, -1] > 0:
            return src_point, True

        dst_point = src_point.copy()

        # case 1: neck and left shoulder are available
        if src_point[17, -1] > 0 and src_point[5, -1] > 0:
            dst_point[6, ] = 2 * src_point[17, ] - src_point[5, ]
            return dst_point, True

        # case 2: right elbow is not available
        if src_point[8, -1] == 0:
            return dst_point, False

        # case 3:
        if src_point[5, -1] > 0 and src_point[7, -1] > 0:
            d = np.sqrt(np.sum((src_point[5, ]-src_point[7, ])**2))
        elif src_point[10, -1] > 0:
            d = np.sqrt(np.sum((src_point[8, ]-src_point[10, ])**2))
        elif src_point[7, -1] > 0 and src_point[9, -1] > 0:
            d = np.sqrt(np.sum((src_point[7, ]-src_point[9, ])**2))
        else:
            return dst_point, False

        depth_frame = rgbd_frame['depth']
        cross = [-1, -1]
        p21 = list(rgbd_kp[8])
        p22 = list(rgbd_kp[6])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[6] = computeP3withD(src_point[8], point_cross, d)

        return dst_point, True

    def gen_lshoulder(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        if src_point[5, -1] > 0:
            return src_point, True

        dst_point = src_point.copy()

        # case 1: neck and left shoulder are available
        if src_point[17, -1] > 0 and src_point[6, -1] > 0:
            dst_point[5, ] = 2 * src_point[17, ] - src_point[6, ]
            return dst_point, True

        # case 2: left elbow is not available
        if src_point[7, -1] == 0:
            return dst_point, False

        # case 3:
        if src_point[6, -1] > 0 and src_point[8, -1] > 0:
            d = np.sqrt(np.sum((src_point[6, ]-src_point[8, ])**2))
        elif src_point[9, -1] > 0:
            d = np.sqrt(np.sum((src_point[7, ]-src_point[9, ])**2))
        elif src_point[8, -1] > 0 and src_point[10, -1] > 0:
            d = np.sqrt(np.sum((src_point[8, ]-src_point[10, ])**2))
        else:
            return dst_point, False

        depth_frame = rgbd_frame['depth']
        cross = [-1, -1]
        p21 = list(rgbd_kp[7])
        p22 = list(rgbd_kp[5])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[5] = computeP3withD(src_point[7], point_cross, d)
        return dst_point, True

    def gen_relbow(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        # case 0: right elbow is already available
        if src_point[8, -1] > 0:
            return src_point, True

        # case 1: right elbow is not available on fisheye camera
        if fish_kp[8, 0] < 1e-5 and fish_kp[8, 1] < 1e-5:
            return src_point, False

        # case 2: right shoulder is not available on rgbd camera
        p_rshoulder = src_point[6]
        if p_rshoulder[-1] == 0:
            return src_point, False

        # case 2: left elbow and left shoulder are available
        # case 3: left elbow and left wrist are available
        p_lshoulder = src_point[5]
        p_lelbow = src_point[7]
        p_lwrist = src_point[9]
        if p_lshoulder[-1] > 0 and p_lelbow[-1] > 0:
            d = np.sqrt(np.sum((p_lshoulder-p_lelbow)**2))
        elif p_lelbow[-1] > 0 and p_lwrist[-1] > 0:
            d = np.sqrt(np.sum((p_lwrist-p_lelbow)**2))
        else:
            return src_point, False

        # case 4:
        depth_frame = rgbd_frame['depth']
        dst_point = src_point.copy()
        cross = [-1, -1]
        p21 = list(rgbd_kp[6])
        p22 = list(rgbd_kp[8])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[8] = computeP3withD(p_rshoulder, point_cross, d)

        return dst_point, True

    def gen_lelbow(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        # case 0: left elbow is already available
        if src_point[7, -1] > 0:
            return src_point, True

        # case 1: left elbow is not available on fisheye camera
        if fish_kp[7, 0] < 1e-5 and fish_kp[7, 1] < 1e-5:
            return src_point, False

        # case 2: left shoulder is not available on rgbd camera
        p_lshoulder = src_point[5]
        if p_lshoulder[-1] == 0:
            return src_point, False

        # case 3: left elbow and left shoulder are available
        # case 4: left elbow and left wrist are available
        p_rshoulder = src_point[6]
        p_relbow = src_point[8]
        p_rwrist = src_point[10]
        if p_rshoulder[-1] > 0 and p_relbow[-1] > 0:
            d = np.sqrt(np.sum((p_rshoulder-p_relbow)**2))
        elif p_relbow[-1] > 0 and p_rwrist[-1] > 0:
            d = np.sqrt(np.sum((p_rwrist-p_relbow)**2))
        else:
            return src_point, False

        # case 4:
        depth_frame = rgbd_frame['depth']
        dst_point = src_point.copy()
        cross = [-1, -1]
        p21 = list(rgbd_kp[5])
        p22 = list(rgbd_kp[7])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[7] = computeP3withD(p_lshoulder, point_cross, d)

        return dst_point, True

    def gen_rwrist(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        '''only for trt-pose
        '''
        # case 0: right wrist is already available
        if src_point[10, -1] > 0:
            return src_point, True

        dst_point = src_point.copy()
        # print("dst_point[8, -1] = ", dst_point[8, -1])
        depth_frame = rgbd_frame['depth']

        p_rshoulder = src_point[6]
        p_relbow = src_point[8]
        p_lelbow = src_point[7]
        p_lwrist = src_point[9]
        p_lshoulder = src_point[5]

        # case 1: right wrist is not available on fisheye camera
        if fish_kp[10, 0] < 1e-5 and fish_kp[10, 1] < 1e-5:
            return dst_point, False

        # case 2: invalid right elbow
        if p_relbow[-1] == 0:
            return dst_point, False

        if p_lelbow[-1] > 0 and p_lwrist[-1] > 0:
            d = np.sqrt(np.sum((p_lelbow-p_lwrist)**2))
        elif p_lelbow[-1] > 0 and p_lshoulder[-1] > 0:
            d = np.sqrt(np.sum((p_lelbow-p_lshoulder)**2))
        elif p_rshoulder[-1] > 0:
            d = np.sqrt(np.sum((p_relbow-p_rshoulder)**2))
        else:
            return dst_point, False

        # case 4:
        cross = [-1, -1]
        p21 = list(rgbd_kp[8])
        p22 = list(rgbd_kp[10])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[10] = computeP3withD(p_relbow, point_cross, d)

        return dst_point, True

    def gen_lwrist(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        '''only for trt-pose
        '''
        # case 0: left wrist is already available
        if src_point[9, -1] > 0:
            return src_point, True

        dst_point = src_point.copy()
        depth_frame = rgbd_frame['depth']

        p_rshoulder = src_point[6]
        p_lelbow = src_point[7]
        p_relbow = src_point[8]
        p_rwrist = src_point[10]
        p_lshoulder = src_point[5]
        p_rshoulder = src_point[6]

        # case 1: left wrist is not available on fisheye camera
        if fish_kp[9, 0] < 1e-5 and fish_kp[9, 1] < 1e-5:
            return dst_point, False

        # case 2: invalid left elbow
        if p_lelbow[-1] == 0:
            return dst_point, False

        if p_relbow[-1] > 0 and p_rwrist[-1] > 0:
            d = np.sqrt(np.sum((p_relbow-p_rwrist)**2))
        elif p_relbow[-1] > 0 and p_rshoulder[-1] > 0:
            d = np.sqrt(np.sum((p_relbow-p_rshoulder)**2))
        elif p_lshoulder[-1] > 0:
            d = np.sqrt(np.sum((p_lelbow-p_lshoulder)**2))
        else:
            return dst_point, False

        # case 4:
        cross = [-1, -1]
        p21 = list(rgbd_kp[7])
        p22 = list(rgbd_kp[9])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[9] = computeP3withD(p_lelbow, point_cross, d)

        return dst_point, True

    def gen_rknee(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        # case 0: right knee is already available
        if src_point[14, -1] > 0:
            return src_point, True

        # case 1: right knee is not available on fisheye camera
        if fish_kp[14, 0] < 1e-5 and fish_kp[14, 1] < 1e-5:
            return src_point, False

        # case 2: right hip is not available
        if src_point[12, -1] == 0:
            return src_point, False

        # case 3:
        if src_point[11, -1] > 0 and src_point[13, -1] > 0:
            d = np.sqrt(np.sum((src_point[11, ]-src_point[13, ])**2))
        elif src_point[17, -1] > 0:
            d = np.sqrt(np.sum((src_point[17, ]-src_point[12, ])**2))
            d1 = np.sqrt(np.sum((fish_kp[14, ]-fish_kp[12, ])**2))
            d2 = np.sqrt(np.sum((fish_kp[17, ]-fish_kp[12, ])**2))
            ratio = min(max(d1/d2, 0.5), 1.0)
            d *= ratio
        else:
            return src_point, False

        dst_point = src_point.copy()
        depth_frame = rgbd_frame['depth']
        cross = [-1, -1]
        p21 = list(rgbd_kp[12])
        p22 = list(rgbd_kp[14])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[14] = computeP3withD(src_point[12], point_cross, d)

        return dst_point, True

    def gen_lknee(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        # case 0: left knee is already available
        if src_point[13, -1] > 0:
            return src_point, True

        # case 1: left knee is not available on fisheye camera
        if fish_kp[13, 0] < 1e-5 and fish_kp[13, 1] < 1e-5:
            return src_point, False

        # case 2: left hip is not available
        if src_point[11, -1] == 0:
            return src_point, False

        # case 3:
        if src_point[12, -1] > 0 and src_point[14, -1] > 0:
            d = np.sqrt(np.sum((src_point[12, ]-src_point[14, ])**2))
        elif src_point[17, -1] > 0:
            d = np.sqrt(np.sum((src_point[17, ]-src_point[11, ])**2))
            d1 = np.sqrt(np.sum((fish_kp[11, ]-fish_kp[13, ])**2))
            d2 = np.sqrt(np.sum((fish_kp[17, ]-fish_kp[11, ])**2))
            ratio = min(max(d1/d2, 0.5), 1.0)
            d *= ratio
        else:
            return src_point, False

        dst_point = src_point.copy()
        depth_frame = rgbd_frame['depth']
        cross = [-1, -1]
        p21 = list(rgbd_kp[11])
        p22 = list(rgbd_kp[13])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[13] = computeP3withD(src_point[11], point_cross, d)

        return dst_point, True

    def gen_rankle(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        # case 0: right ankle is already available
        if src_point[16, -1] > 0:
            return src_point, True

        # case 1: right ankle is not available on fisheye camera
        if fish_kp[16, 0] < 1e-5 and fish_kp[16, 1] < 1e-5:
            return src_point, False

        # case 2: right knee is not available
        if src_point[14, -1] == 0:
            return src_point, False

        # case 3:
        if src_point[13, -1] > 0 and src_point[15, -1] > 0:
            d = np.sqrt(np.sum((src_point[13, ]-src_point[15, ])**2))
        elif src_point[12, -1] > 0:
            d = np.sqrt(np.sum((src_point[12, ]-src_point[14, ])**2))
        elif src_point[11, -1] > 0 and src_point[13, -1] > 0:
            d = np.sqrt(np.sum((src_point[11, ]-src_point[13, ])**2))
        else:
            return src_point, False

        dst_point = src_point.copy()
        depth_frame = rgbd_frame['depth']
        cross = [-1, -1]
        p21 = list(rgbd_kp[14])
        p22 = list(rgbd_kp[16])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[16] = computeP3withD(src_point[14], point_cross, d)

        return dst_point, True

    def gen_lankle(self, src_point, rgbd_frame, fish_kp, rgbd_kp):
        # case 0: left ankle is already available
        if src_point[15, -1] > 0:
            return src_point, True

        # case 1: left ankle is not available on fisheye camera
        if fish_kp[15, 0] < 1e-5 and fish_kp[15, 1] < 1e-5:
            return src_point, False

        # case 2: left knee is not available
        if src_point[13, -1] == 0:
            return src_point, False

        # case 3:
        if src_point[14, -1] > 0 and src_point[16, -1] > 0:
            d = np.sqrt(np.sum((src_point[14, ]-src_point[16, ])**2))
        elif src_point[11, -1] > 0:
            d = np.sqrt(np.sum((src_point[11, ]-src_point[13, ])**2))
        elif src_point[12, -1] > 0 and src_point[14, -1] > 0:
            d = np.sqrt(np.sum((src_point[12, ]-src_point[14, ])**2))
        else:
            return src_point, False

        dst_point = src_point.copy()
        depth_frame = rgbd_frame['depth']
        cross = [-1, -1]
        p21 = list(rgbd_kp[13])
        p22 = list(rgbd_kp[15])
        for p11, p12 in self.corners:
            cross = cross_point_in(p11, p12, p21, p22, self.size)
            if cross[0] >= 0 and cross[1] >= 0:
                break
        if cross == [-1, -1]:
            return dst_point, False
        cross = tuple(map(int, cross))
        point_cross = self.extractor.pixel_to_point(depth_frame, cross)
        dst_point[15] = computeP3withD(src_point[13], point_cross, d)

        return dst_point, True
