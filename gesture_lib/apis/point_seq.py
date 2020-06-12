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


def get_face_angle(point, mode='openpose'):
    angle = np.zeros([3, ])
    if mode == 'openpose':
        nose = point[0, :3]
        r_eye = point[15, :3]
        l_eye = point[16, :3]
    elif mode == 'trtpose':
        nose = point[0, :]
        r_eye = point[2, :]
        l_eye = point[1, :]
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


def get_body_angle(point, mode='openpose'):
    angle = np.zeros([3, ])
    if mode == 'openpose':
        r_shoulder = point[2, :3]
        l_shoulder = point[5, :3]
        midhip = point[8, :3]
    elif mode == 'trtpose':
        r_shoulder = point[6, :]
        l_shoulder = point[5, :]
        midhip = np.mean(point[11:13, :], axis=0)
    else:
        raise
    dis = np.sqrt(np.sum((l_shoulder-r_shoulder)**2))

    if dis < 0.1:
        print("small dis")
        return angle
    elif r_shoulder[-1] * l_shoulder[-1] * midhip[-1] == 0:
        print("loss points info for compute body angle")
        return angle
    else:
        v1 = r_shoulder[:3] - midhip[:3]
        v2 = l_shoulder[:3] - midhip[:3]
        norm_v = np.cross(v1, v2)
        norm_d = np.sqrt(np.sum(norm_v ** 2)) + 1e-6
        angle = np.arccos(norm_v / norm_d) * 180 / np.pi
        return angle


def get_shoulder_hip_dis(point, mode='openpose'):
    if mode == "openpose":
        r_shoulder = point[2, :]
        l_shoulder = point[5, :]
        midhip = point[8, :]
    elif mode == 'trtpose':
        r_shoulder = point[6, :]
        l_shoulder = point[5, :]
        midhip = np.mean(point[11:13, :], axis=0)
    else:
        raise

    dis = np.sqrt(np.sum((l_shoulder[:3]-r_shoulder[:3])**2))
    dis += np.sqrt(np.sum((l_shoulder[:3]-midhip[:3])**2))
    dis += np.sqrt(np.sum((r_shoulder[:3]-midhip[:3])**2))
    return dis


def get_location(point):
    valid_point = point[point[:, 2] > 0, :3]
    if valid_point.shape[0] == 0:
        return np.zeros([3])
    else:
        valid_point = remove_outlier(valid_point, idx_list=[2])
        return np.mean(valid_point, axis=0)


def get_keypoint_box(keypoint):
    keypoint_box = np.zeros(4)  # [x1, y1, x1, y2]
    valid_keypoint = keypoint[keypoint[:, 0] > 0, :2]
    keypoint_num = valid_keypoint.shape[0]
    if keypoint_num == 0:
        return keypoint_box
    elif keypoint_num == 1:
        keypoint_box[:2] = valid_keypoint[0, :2]
        keypoint_box[2:] = valid_keypoint[0, :2]
        return keypoint_box
    else:
        keypoint_box[:2] = np.min(valid_keypoint, axis=0)
        keypoint_box[2:] = np.max(valid_keypoint, axis=0)
        return keypoint_box


def box_iou(box1, box2, mode='iou'):
    assert mode in ['iou', 'diou']
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    area1 = (x12-x11+1) * (y12-y11+1)
    area2 = (x22-x21+1) * (y22-y21+1)

    xx1, xx2 = max(x11, x21), min(x12, x22)
    yy1, yy2 = max(y11, y21), min(y12, y22)
    inter = max(0, xx2-xx1+1) * max(0, yy2-yy1+1)

    iou = inter / (area1 + area2 - inter)

    if mode == 'iou':
        return iou
    elif mode == 'diou':
        c1_x, c1_y = (x11 + x12) / 2, (y11 + y12) / 2
        c2_x, c2_y = (x21 + x22) / 2, (y21 + y22) / 2
        d = (c1_x-c2_x) ** 2 + (c1_y-c2_y) ** 2
        xx1, xx2 = min(x11, x21), max(x12, x22)
        yy1, yy2 = min(y11, y21), max(y12, y22)
        c = (xx2-xx1) ** 2 + (yy2-yy1) ** 2
        diou = iou - d / c
        return diou
    else:
        raise


def get_max_diou_skeleton(target_box, skeletons):
    diou_list = []
    for keypoint, _ in skeletons:
        current_box = get_keypoint_box(keypoint)
        current_diou = box_iou(target_box, current_box)
        diou_list.append(current_diou)
    idx = np.argmax(diou_list)
    return skeletons[idx]




if __name__ == "__main__":
    array = np.arange(24).reshape([2, 12])
    result = get_keypoint_box(array)
    print(result)
