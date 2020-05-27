import numpy as np


def is_static(points_array):
    ''' get static state based on the points array variance
    '''
    var_mean = np.mean(np.var(points_array, axis=0))
    var_max = np.max(np.var(points_array, axis=0))
    # print("var mean = {}, var max = {}".format(var_mean, var_max))
    if var_max < 0.001 and var_mean < 0.0001:
        # print("static: var_max = {}, var_mean = {}".format(var_max, var_mean))
        return True
    else:
        return False


def post_process_wave(points_array, mode='openpose'):
    '''futher assertion for wave

    Args:
        points_array: x,y,z info of 'Neck', 'RShoulder', 'RElbow', 'RWrist', (T, 12)

    Return:
        True / False: wave or not
    '''
    print("in post procss: wave")
    if mode == "openpose":
        points_array = remove_outlier(points_array, [7, 9, 10, 11])
        rwrist_x = points_array[:, 9]
        rwrist_y = points_array[:, 10]
        rwrist_z = points_array[:, 11]
        relbow_y = points_array[:, 7]
    elif mode == "trtpose":
        points_array = remove_outlier(points_array, [4, 6, 7, 8])
        rwrist_x = points_array[:, 6]
        rwrist_y = points_array[:, 7]
        rwrist_z = points_array[:, 8]
        relbow_y = points_array[:, 4]
    else:
        raise
    rwrist_x = rwrist_x[rwrist_x != 0]
    rwrist_y = rwrist_y[rwrist_y != 0]
    rwrist_z = rwrist_z[rwrist_z != 0]
    relbow_y = relbow_y[relbow_y != 0]

    if rwrist_x.shape[0] * rwrist_y.shape[0] * rwrist_z.shape[0] * \
       relbow_y.shape[0] == 0:
        print("wave: shape 0")
        return False

    if np.mean(rwrist_y) >= np.mean(relbow_y):
        print("wave: too high relbow_y")
        return False

    if np.max(rwrist_x) - np.min(rwrist_x) < 0.05:
        print("wave: too small rwrist_x")
        return False

    if np.max(rwrist_z) - np.min(rwrist_z) > 0.25:
        print("wave: too large rwrist_z")
        return False

    return True


def post_process_come(points_array, mode='openpose'):
    '''futher assertion for come

    Args:
        points_array: x,y,z info of 'Neck', 'RShoulder', 'RElbow', 'RWrist', (T, 12)

    Return:
        True / False: come or not
    '''
    print("in post procss: come")
    if mode == "openpose":
        points_array = remove_outlier(points_array, [4, 7, 9, 10, 11])
        rwrist_x = points_array[:, 9]
        rwrist_y = points_array[:, 10]
        rwrist_z = points_array[:, 11]
        relbow_y = points_array[:, 7]
        rshoulder_y = points_array[:, 4]
    elif mode == "trtpose":
        points_array = remove_outlier(points_array, [1, 4, 6, 7, 8])
        rwrist_x = points_array[:, 6]
        rwrist_y = points_array[:, 7]
        rwrist_z = points_array[:, 8]
        relbow_y = points_array[:, 4]
        rshoulder_y = points_array[:, 1]
    else:
        raise

    rwrist_x = rwrist_x[rwrist_x != 0]
    rwrist_y = rwrist_y[rwrist_y != 0]
    rwrist_z = rwrist_z[rwrist_z != 0]
    relbow_y = relbow_y[relbow_y != 0]
    rshoulder_y = rshoulder_y[rshoulder_y != 0]

    if rwrist_x.shape[0] * rwrist_y.shape[0] * rwrist_z.shape[0] * \
       relbow_y.shape[0] * rshoulder_y.shape[0] == 0:
        return False
    if np.mean(rwrist_y) - np.mean(relbow_y) > 0.05:
        print("come: too low relbow")
        return False
    if np.mean(rshoulder_y) >= np.mean(relbow_y):
        print("come: too high relbow")
        return False
    if np.max(rwrist_x) - np.min(rwrist_x) > 0.18:
        print("come: too large rwrist_x")
        return False
    if np.max(rwrist_z) - np.min(rwrist_z) < 0.1:
        print("come: too small rwrist_z")
        return False

    return True


def post_process_hello(points_array, mode='openpose'):
    '''futher assertion for hello

    Args:
        points_array: x,y,z info of 'Neck', 'RShoulder', 'RElbow', 'RWrist', (T, 12)

    Return:
        True / False: come or not
    '''
    print("in post procss: hello")
    if mode == "openpose":
        points_array = remove_outlier(points_array, [1, 10])
        rwrist_y = points_array[:, 10]
        neck_y = points_array[:, 1]
    elif mode == "trtpose":
        points_array = remove_outlier(points_array, [7, 10])
        rwrist_y = points_array[:, 7]
        neck_y = points_array[:, 10]
    else:
        raise

    rwrist_y = rwrist_y[rwrist_y != 0]
    neck_y = neck_y[neck_y != 0]
    num1 = rwrist_y.shape[0] // 10
    num2 = neck_y.shape[0] // 10

    if rwrist_y.shape[0] * neck_y.shape[0] == 0:
        return False

    if num1 < 1 or num2 < 1:
        return False

    if rwrist_y.shape[0] < points_array.shape[0] // 3:
        return False

    if np.mean(rwrist_y[0:num1] - np.mean(rwrist_y[-num1:])) < 0.3:
        print("hello: too low for hello")
        return False

    if np.mean(rwrist_y[-num1:]) > np.mean(neck_y[-num2:]):
        print("hello: lower than neck")
        return False

    return True


def get_face_angle(points_array, mode='openpose'):
    angle = np.zeros([3, ])
    if mode == 'openpose':
        nose = points_array[0, :3]
        r_eye = points_array[15, :3]
        l_eye = points_array[16, :3]
    elif mode == 'trtpose':
        nose = points_array[0, :]
        r_eye = points_array[2, :]
        l_eye = points_array[1, :]
    else:
        raise

    if nose[-1] * r_eye[-1] * l_eye[-1] == 0:
        print("loss keypoints info for compute face angle")
        return angle
    else:
        v1 = r_eye[:3] - nose[:3]
        v2 = l_eye[:3] - nose[:3]
        norm_v = np.cross(v1, v2)
        norm_d = np.sqrt(np.sum(norm_v ** 2)) + 1e-6
        angle = np.arccos(norm_v / norm_d) * 180 / np.pi
        return angle


def get_body_angle(points_array, mode='openpose'):
    angle = np.zeros([3, ])
    if mode == 'openpose':
        r_shoulder = points_array[2, :3]
        l_shoulder = points_array[5, :3]
        midhip = points_array[8, :3]
    elif mode == 'trtpose':
        r_shoulder = points_array[6, :]
        l_shoulder = points_array[5, :]
        midhip = np.mean(points_array[11:13, :], axis=0)
    else:
        raise
    dis = np.sqrt(np.sum((l_shoulder-r_shoulder)**2))

    if dis < 0.1:
        print("small dis")
        return angle
    elif r_shoulder[-1] * l_shoulder[-1] * midhip[-1] == 0:
        print("loss keypoints info for compute body angle")
        return angle
    else:
        v1 = r_shoulder[:3] - midhip[:3]
        v2 = l_shoulder[:3] - midhip[:3]
        norm_v = np.cross(v1, v2)
        norm_d = np.sqrt(np.sum(norm_v ** 2)) + 1e-6
        angle = np.arccos(norm_v / norm_d) * 180 / np.pi
        return angle


def get_shoulder_hip_dis(points_array, mode='openpose'):
    if mode == "openpose":
        r_shoulder = points_array[2, :]
        l_shoulder = points_array[5, :]
        midhip = points_array[8, :]
    elif mode == 'trtpose':
        r_shoulder = points_array[6, :]
        l_shoulder = points_array[5, :]
        midhip = np.mean(points_array[11:13, :], axis=0)
    else:
        raise

    dis = np.sqrt(np.sum((l_shoulder[:2]-r_shoulder[:2])**2))
    dis += np.sqrt(np.sum((l_shoulder[:2]-midhip[:2])**2))
    dis += np.sqrt(np.sum((r_shoulder[:2]-midhip[:2])**2))
    return dis


def remove_outlier(points_array, idx_list=None):
    T, D = points_array.shape
    if idx_list is None:
        idx_list = range(D)
    temp_array = points_array[:, idx_list]
    temp_mean = np.mean(temp_array, axis=0)
    temp_std = np.std(temp_array, axis=0)
    mask1 = temp_array > (temp_mean - 1.5*temp_std)
    mask2 = temp_array < (temp_mean + 1.5*temp_std)
    mask = np.prod(mask1 * mask2, axis=1) > 0
    result_array = points_array[mask]
    return result_array


if __name__ == "__main__":
    array = np.arange(24).reshape([8, 3])
    result = remove_outlier(array, [1])
    print(result)
