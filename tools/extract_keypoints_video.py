# Description: openpose body keypoints detection and extraction for video
# Author: ZhouJH
# Data: 2020/4/7

import sys
sys.path.append(".")

import numpy as np
import cv2
from tqdm import tqdm
from utils import get_file_path
import pyopenpose as op


def _get_max_score_keypoints(pose_keypoints):
    '''get the body keypoints with maximum score in one frame

    for keypoints extraction
    
    Args:
        pose_keypoints: [ndarray], shape (N, 25, 3)

    Return:
        pose_keypoint: [ndarray], shape (25, 3)
    '''
    if len(pose_keypoints.shape) == 0:
        return np.zeros([25, 3])

    scores_array = np.mean(pose_keypoints[:,:,-1], axis=-1)
    best_idx = np.argmax(scores_array)
    return pose_keypoints[best_idx, :, :]


def body_keypoints_video(op_wrapper, datum, video_path, show=True, save_best=None):
    '''detect body keypoints

    Args:
        op_wrapper: [op.WrapperPython()], body keypoints detector
        datum: [op.Datum()], saveing input and result
        video_path: [str] image path for detecting
        save_best: [str | None], txt path for saveing best body keypoints

    Return:
        None
    '''

    best_keypoint_list = []
    
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            datum.cvInputData = frame
            op_wrapper.emplaceAndPop([datum])
            #print(datum.poseKeypoints.shape)
            best_keypoint = _get_max_score_keypoints(datum.poseKeypoints)
            best_keypoint[:,0] = best_keypoint[:,0] / frame.shape[1]
            best_keypoint[:,1] = best_keypoint[:,1] / frame.shape[0]
            if best_keypoint.any() > 0:
                best_keypoint_list.append(best_keypoint.flatten())
                
            if show is True:
                cv2.imshow("result", datum.cvOutputData)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        else:
            break
    
    cap.release()
    if show is True:
        cv2.destroyAllWindows()

    if save_best is not None:
        try:
            np.savetxt(save_best, np.array(best_keypoint_list), fmt='%.5f')
            print("save bets keypoints to : {}".format(save_best))
        except Exception as e:
            print("save bets keypoints failed")
            print(e)


def body_keypoints_video_list(op_wrapper, datum, video_dir, ext='.mp4'):
    '''
    '''
    video_paths = get_file_path(video_dir, ext)
    for video_path in tqdm(video_paths):
        print("processing video: {}".format(video_path))
        save_best = video_path.replace(ext, '.txt')
        body_keypoints_video(op_wrapper, datum, video_path, False, save_best)
        



if __name__ == "__main__":
    # # process single video
    # video_path = "/home/zhoujh/Data/gesture_recognition/src_data/examples/example_240p.mp4"
    # model_folder = "/home/zhoujh/Project/openpose/models"
    # #save_best = "/home/zhoujh/Data/gesture_recognition/src_data/train/other/other6.txt"

    # params = dict()
    # params["model_folder"] = model_folder
    # op_wrapper = op.WrapperPython()
    # op_wrapper.configure(params)

    # op_wrapper.start()

    # datum = op.Datum()
    # body_keypoints_video(op_wrapper, datum, video_path, show=True)

    ## process video list
    video_dir = "/home/zhoujh/Data/gesture_recognition/src_data/"
    model_folder = "/home/zhoujh/Project/openpose/models"

    params = dict()
    params["model_folder"] = model_folder
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)
    op_wrapper.start()
    datum = op.Datum()

    body_keypoints_video_list(op_wrapper, datum, video_dir, ext='.mp4')



    