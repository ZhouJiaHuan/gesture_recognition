# Description: openpose body keypoints detection for image
# Author: ZhouJH
# Data: 2020/4/7

import sys
sys.path.append(".")
import pyopenpose as op
import cv2

def body_keypoints(op_wrapper, datum, img_path, out_img=None):
    '''detect body keypoints

    Args:
        op_wrapper: [op.WrapperPython()], body keypoints detector
        datum: [op.Datum()], saveing input and result
        img_path: [str] image path for detecting
        out_img: [None | str], output result path
    '''
    img_process = cv2.imread(img_path)
    if img_process is None:
        return []

    datum.cvInputData = img_process
    op_wrapper.emplaceAndPop([datum])
    
    if out_img is not None:
        try:
            cv2.imwrite(out_img, datum.cvOutputData)
            print("save result to {}".format(out_img))
        except Exception as e:
            print("saved failed!")
            print(e)
        
    return datum.poseKeypoints
    
     
    
if __name__ == "__main__":
    img_path = "/home/zhoujh/Project/gesture_recognition/temp.jpg"
    out_img = "./result.jpg"
    model_folder = "/home/zhoujh/Project/openpose/models"
    params = dict()
    params["model_folder"] = model_folder
    op_wrapper = op.WrapperPython()
    op_wrapper.configure(params)

    op_wrapper.start()

    datum = op.Datum()
    pose_keypoints = body_keypoints(op_wrapper, datum, img_path, out_img)
    print("pose_keypoint shape: {}".format(pose_keypoints.shape))
    print(pose_keypoints)


    