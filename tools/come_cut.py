# crop the gesture of hello

import numpy as np 
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt
import glob
import sys
sys.path.append(".") # relative path of openpose

from tqdm import tqdm
from utils import make_dirs

from dataset.transforms import BodyZeroInterpolation

def smooth_filter(vector):
    L = vector.shape[0]
    vector_filtered = vector.copy()
    for i in range(1, L-1):
        vector_filtered[i] = 0.5*vector[i] + 0.25*vector[i-1] + 0.25*vector[i+1]
    return vector_filtered

def trunc_vector(vector):
    vector_trunc = vector.copy()
    max_thr = min(vector) + 0.6*(max(vector)-min(vector))
    min_thr = min(vector) + 0.4*(max(vector)-min(vector))
    mean_thr = np.mean(vector_trunc)
    vector_trunc[vector>mean_thr] = np.max(vector)
    vector_trunc[vector<mean_thr] = np.min(vector)
    return vector_trunc


def arg_relmaxmin(vector, ignore_s=60, ignore_e=60, min_gap=40):
    rel_min = []
    rel_max = []
    L = len(vector)

    vector_trunc = smooth_filter(vector)
    vector_trunc = trunc_vector(vector_trunc)

    latest_max = 0
    latest_min = 0
    latest_idx = 0
    for i, val in enumerate(vector_trunc):
        if (i < ignore_e):
            continue
        if (i-latest_idx) < min_gap:
            continue

        if val == 0:
            continue

        if (val < vector_trunc[i-1]) and (val == vector_trunc[i+1]) and (i-latest_max > 30):
            rel_max.append(i)
            latest_max = i
            latest_idx = i
        if (val == vector_trunc[i-1]) and (val < vector_trunc[i+1]) and (i-latest_min > 30):
            rel_min.append(i)
            latest_min = i
            latest_idx = i

        if i > L - ignore_e:
            break

    return rel_min, rel_max

def split_points(points, rel_min, rel_max, txt_dir):
    min_array = np.array(rel_min)
    max_array = np.array(rel_max)

    for i, ma in enumerate(max_array[:-1]):
        if np.max(min_array) > ma:
            mi = np.min(min_array[min_array>ma])
            if mi > max_array[i+1]:
                continue
            temp_array = points[ma:mi, :]
            txt_name = "come_" + str(i) + '-' + str(ma) + '-' + str(mi) + '.txt'
            txt_path = os.path.join(txt_dir, txt_name)
            np.savetxt(txt_path, temp_array, fmt='%.5f')

    for i, mi in enumerate(min_array[:-1]):
        if np.max(max_array) > mi:
            ma = np.min(max_array[max_array>mi])
            if ma > min_array[i+1]:
                continue
            temp_array = points[mi:ma, :]
            txt_name = "others_" + str(i) + '-' + str(mi) + '-' + str(ma) + '.txt'
            txt_path = os.path.join(txt_dir, txt_name)
            np.savetxt(txt_path, temp_array, fmt='%.5f')

def split(txt_dir, out_dir, col, ext='op-25.txt'):
    txt_path_list = glob.glob(os.path.join(txt_dir, '*'+ext))
    for txt_path in tqdm(txt_path_list):
        print("process {}".format(txt_path))
        txt_name = os.path.basename(txt_path).split('.txt')[0]
        sub_dir = os.path.join(out_dir, txt_name)
        make_dirs(sub_dir)
        points = np.loadtxt(txt_path)
        points = BodyZeroInterpolation(d=3, index_list=[col//3])(points)
        vector = points[:,col]
        rel_min, rel_max = arg_relmaxmin(vector)
        print(rel_min, rel_max)
        plt.figure(figsize=(100, 40))
        plt.plot(vector)
        for peak in rel_min:
            plt.vlines(peak, -1, 1, colors='r', linestyles='--')
        for peak in rel_max:
            plt.vlines(peak, -1, 1, colors='g', linestyles='--')
        plt.grid()
        plt.savefig(os.path.join(sub_dir, 'vector.jpg'))
        split_points(points, rel_min, rel_max, sub_dir)
        
 
if __name__ == "__main__":
    come_dir = "/home/zhoujh/Data/gesture_recognition/bag_files/come"
    out_dir = "/home/zhoujh/Data/gesture_recognition/bag_files/come_out"
    # split(come_dir, out_dir, col=17)  # rhand y for openpose-body-25
    split(come_dir, out_dir, col=31, ext='*trt-18.txt')  # rhand y for trt-pose-body-18
