# crop the gesture of hello

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import sys
sys.path.append(".")
from gesture_lib.ops.io import make_dirs


def smooth_filter(vector):
    L = vector.shape[0]
    vector_filtered = vector.copy()
    for i in range(1, L):
        vector_filtered[i] = 0.5*vector[i] + 0.5*vector[i-1]
    return vector_filtered

def trunc_vector(vector):
    vector_trunc = vector.copy()
    sorted_vector = np.sort(vector_trunc)[10:-10]
    max_thr = min(sorted_vector) + 0.7*(max(sorted_vector)-min(sorted_vector))
    min_thr = min(sorted_vector) + 0.7*(max(sorted_vector)-min(sorted_vector))
    mean_thr = np.mean(np.sort(vector_trunc)[15:-15])
    vector_trunc[vector>max_thr] = np.max(vector)
    vector_trunc[vector<min_thr] = np.min(vector)
    return vector_trunc


def arg_relmaxmin(vector, ignore_s=60, ignore_e=60, min_gap=30):
    rel_min = []
    rel_max = []
    L = len(vector)

    # vector_trunc = smooth_filter(vector)
    vector_trunc = trunc_vector(vector)

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

        if (val == vector_trunc[i-1]) and (val > vector_trunc[i+1]) and (i-latest_max > 20):
            rel_max.append(i-1)
            latest_max = i
            latest_idx = i
        if (val == vector_trunc[i-1]) and (val < vector_trunc[i+1]) and (i-latest_min > 20):
            rel_min.append(i-1)
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
            txt_name = "hello_" + str(i) + '-' + str(ma) + '-' + str(mi) + '.txt'
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
    for txt_path in txt_path_list:
        print("process {}".format(txt_path))
        txt_name = os.path.basename(txt_path).split('.txt')[0]
        sub_dir = os.path.join(out_dir, txt_name)
        make_dirs(sub_dir)
        points = np.loadtxt(txt_path)
        # points = smooth_filter(points)
        vector = points[:,col]
        # vector = trunc_vector(vector)
        rel_min, rel_max = arg_relmaxmin(vector)
        print(rel_min, rel_max)
        if len(rel_min) * len(rel_max) == 0:
            print("parse failed for {}".format(txt_path))
            continue
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
    hello_dir = "/home/zhoujh/Data/gesture_recognition/bag_files/hello"
    out_dir = "/home/zhoujh/Data/gesture_recognition/bag_files/hello_out"
    # split(hello_dir, out_dir, col=17) # hand y
    split(hello_dir, out_dir, col=31, ext='*trt-18.txt')  # rhand y for trt-pose-body-18
