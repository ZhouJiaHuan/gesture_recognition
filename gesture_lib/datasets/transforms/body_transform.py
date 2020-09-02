import numpy as np
import random

from gesture_lib.registry import PIPELINES


@PIPELINES.register_module(name="BodyNormalize")
class BodyNormalize(object):
    '''normalize the body keypoints
    '''

    def __init__(self):
        self.d = 3

    def __call__(self, points_array):
        '''
        Args:
            points_array: [np.array], points array to be normalized

        Return:
            normalized points array
        '''

        T, D = points_array.shape
        d = self.d
        assert D % self.d == 0, "invalid dimentions of keypoints, got {}".format(D)
        num_points = D // d
        result_array = np.zeros([T, D])
        for i in range(num_points):
            L = points_array[:, i*d]**2 + points_array[:, i*d+1]**2 + points_array[:, i*d+2]**2
            L = np.sqrt(L) + 1e-7
            temp = points_array[:, i*d:(i+1)*d] / L.reshape([-1, 1])
            result_array[:, i*d:(i+1)*d] = temp

        return result_array


@PIPELINES.register_module(name="BodyCoordTransform")
class BodyCoordTransform(object):
    '''transform the coordinate
    '''

    def __init__(self):
        self.d = 3

    def __call__(self, points_array):
        origin = points_array[0, :self.d]
        T, D = points_array.shape
        result_array = np.zeros([T, D])
        assert D % self.d == 0, "invalid dimentions of keypoints, got {}".format(D)
        num_points = D // self.d

        for i in range(num_points):
            temp = points_array[:, i*self.d:(i+1)*self.d] - origin
            result_array[:, i*self.d:(i+1)*self.d] = temp
        return result_array


@PIPELINES.register_module(name="BodyInterpolation")
class BodyInterpolation(object):
    '''interpolate the points array when array length < min_n
    '''

    def __init__(self, min_n):
        self.min_n = min_n

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"

        if T >= self.min_n:
            return points_array

        x = np.arange(T)
        xvals = np.linspace(0, T, num=self.min_n)
        result_array = np.zeros([self.min_n, D])
        for i in range(D):
            result_array[:, i] = np.interp(xvals, x, points_array[:, i])

        return result_array


@PIPELINES.register_module(name="BodyZeroInterpolation")
class BodyZeroInterpolation(object):
    '''interpolate the location where the points value is zero.
    '''

    def __init__(self):
        self.d = 3

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"
        result_array = points_array.copy()

        x_eval = np.arange(T)
        for i in range(D):
            temp_array = result_array[:, i]
            mask = temp_array != 0
            keep_x = x_eval[mask]
            keep_array = temp_array[mask]
            if len(keep_x) == T or len(keep_x) < T // 3:
                continue
            result_array[:, i] = np.interp(x_eval, keep_x, keep_array)

        return result_array


@PIPELINES.register_module(name="BodyRandomToZero")
class BodyRandomToZero(object):
    ''' randomly set the points to zero
    '''

    def __init__(self, p=0.05):
        self.d = 3
        self.p = p

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"
        body_num = D // self.d
        p_array = np.random.random([T, body_num])
        p_array[p_array > self.p] = 1
        p_array[p_array <= self.p] = 0
        p_array[0, :] = 1
        p_array[-1, :] = 1

        result_array = points_array.copy()
        for i in range(body_num):
            temp_p = p_array[:, i].reshape([-1, 1])
            temp = result_array[:, i*self.d:(i+1)*self.d] * temp_p
            result_array[:, i*self.d:(i+1)*self.d] = temp

        return result_array


@PIPELINES.register_module(name="BodyOutlierToZero")
class BodyOutlierToZero(object):
    '''detect and set the outlier points to zero.
    '''
    def __init__(self, alpha=1.5):
        self.d = 3
        self.alpha = alpha

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"

        result_array = points_array.copy()
        for i in range(D):
            temp_array = result_array[:, i]
            temp_array = temp_array[temp_array != 0]
            if temp_array.shape[0] == 0:
                continue
            temp_mean = np.mean(temp_array)
            temp_std = np.std(temp_array)
            mask1 = result_array[:, i] > temp_mean + self.alpha * temp_std
            mask2 = result_array[:, i] < temp_mean - self.alpha * temp_std
            mask = mask1 + mask2
            result_array[mask, i] = 0
        return result_array


@PIPELINES.register_module(name="BodyRandomCropFixLen")
class BodyRandomCropFixLen(object):
    '''randomly crop the points array
    '''

    def __init__(self, ratio):
        assert ratio > 0 and ratio < 1
        self.ratio = ratio

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"

        T1 = int(T * self.ratio)
        idx_s = int(np.floor((1-self.ratio) * T))
        result_array = points_array[idx_s:idx_s+T1, :]
        return result_array


@PIPELINES.register_module(name="BodyRandomCropVariable")
class BodyRandomCropVariable(object):
    '''randomly crop the points array
    '''

    def __init__(self, ratio_s, ratio_e):
        assert ratio_s + ratio_e < 1
        self.ratio_s = ratio_s
        self.ratio_e = ratio_e

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"

        idx_s = random.randint(0, int(T * self.ratio_s))
        idx_e = random.randint(int((1-self.ratio_e)*T), T)

        result_array = points_array[idx_s:idx_e, :]
        return result_array


@PIPELINES.register_module(name="BodyRandomSample")
class BodyRandomSample(object):
    '''randomly sample the points array
    '''

    def __init__(self, max_interval=3):
        assert max_interval > 1
        self.max_interval = int(np.ceil(max_interval))

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"
        interval = random.randint(1, self.max_interval)
        result_array = points_array[::interval, :]
        return result_array


@PIPELINES.register_module(name="BodyGaussianNoise")
class BodyGaussianNoise(object):
    '''add gaussian noise to the points info
    '''

    def __init__(self, scale=0.001):
        self.scale = scale

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"
        result_array = np.random.normal(points_array, self.scale)
        return result_array


@PIPELINES.register_module(name="BodyExpSmooth")
class BodyExpSmooth(object):
    '''smooth the points array with exponential smooth method
    '''

    def __init__(self, alpha=0.6):
        assert alpha > 0 and alpha < 1
        self.alpha = alpha

    def __call__(self, points_array):
        T, D = points_array.shape
        assert D > 0, "dimensions should greater than 0!"
        result_array = np.zeros_like(points_array)
        result_array[0,:] = points_array[0,:]
        for i in range(1, T):
            result_array[i, :] = self.alpha*points_array[i,:] + (1-self.alpha)*result_array[i-1,:]

        return result_array


@PIPELINES.register_module(name="BodyResize")
class BodyResize(object):
    '''resize the points array to fixed length
    '''

    def __init__(self, length):
        assert length > 0
        self.length = length

    def __call__(self, points_array):
        T, D = points_array.shape

        result_array = points_array.copy()
        assert D > 0, "dimensions should greater than 0!"
        if T < self.length:
            x = np.arange(T)
            xvals = np.linspace(0, T, num=self.length)
            temp_array = np.zeros([self.length, D])
            for i in range(D):
                temp_array[:, i] = np.interp(xvals, x, points_array[:, i])

            result_array = temp_array
        interval = int(np.floor(result_array.shape[0]/self.length))
        idx = np.arange(self.length) * interval
        result_array = result_array[idx, :]

        return result_array
