This repository realize the simple gestures recognition based on skeletons information extracted by Openpose or Trt-pose. The main process includes 5 key steps

- get a sequence of frames from the depth camera (eg., realsense), including the color frame and depth frame
- extract the skeletons informations with `openpose body-25` model or `trt-pose body-18` model.
- track person with specified feature and DIoU. using features extracted with `SURF` for higher speed and features extracted with dlib face module for higher accuracy.
- parse the 3-d location of the 25 body key points from realsense camera
- predict the gestures from fixed length of frames (eg., 30 frames) with LSTM model

The processed dataset for training and models for face recognition and pose estimation can be downloaded from here:

LINK：https://pan.baidu.com/s/1_6hoD4pP1g5nIyQfhe1xig 
PASSWORD: bezc

# Dependence

- python 3.6
- OpenCV 3.4.2 (with extra modules for `SURF` feature)
- PyTorch 1.4 (with torchvision)
- openpose library
- realsense library
- Dlib library 
- TensorRT: https://github.com/NVIDIA/TensorRT
- torch2trt: https://github.com/NVIDIA-AI-IOT/torch2trt
- trt-pose: https://github.com/NVIDIA-AI-IOT/trt_pose

The `TensorRT` and `torch2trt` are used for converting and loading `trt-pose` model.

# Dataset

The dataset for training and test includes 4 simple gestures, namely, `wave`, `come`, `hello` and `others`. The source samples are collected by realsense and saved as `.bag` files which consist of the color frame and depth frame information.

For LSTM training, body key points information are extracted from the color image in bag file and the 3-D locations of the 25 key points are computed with realsense api. Eventually, each sample for training and test is represented by a T * 100 or T * 75 matrix saved in a txt file, where T is the number of frames related to a gesture and 100 = 25 * (3+1) , namely the x, y, z (meters) with scores of the 25 key points for openpose model. For trt-pose, 75 = 25 * 3, namely the x, y, z without scores.

# Model

We use the `openpose body-25` model (**higher accuracy**) and `trt-pose body-18` model (**higher speed**) for human pose estimation.

The model for gesture recognition is a `LSTM` structure,  which generally includes 2 or 3 LSTM layers and 1 fully connected layer with the SoftMax for classification.  We also add the attention strategy to LSTM in our experiments. See `models/lstm.py` and `models/lstm_attention.py`for more details about the model.

# Train

For training the model, just run:

```bash
python tools/train.py CONFIGURE_FILE
```

the configure file is specified by a `.yaml` file, see example in `configs/lstm_op-body25.yaml` and `configs/lstm_trt-body18.yaml`  for more details.

# Inference

For the inference with realsense camera, run:

```
python tools/camera_inference.py CONFIG CHECKPOINT --show                    
```

For specified example, see `inference_trt.sh`  for inference with the `trt-pose`. If you want to using `openpose` or modify other configures, please refer to `apis/inference.yaml` 

# TODO list

- speed up with Cython or Pybind11 (see gesture_lib/ops/clib and gesture_lib/ops/pybind11_lib/ for part of this work)
- local search algorithm optimization for skeletons mapping from fisheye camera to RGB-D camera
- one more stable matching algorithm for tracking (we use the face recognition and SURF feature for now)
- gesture data collection for more robust gesture recognition (need much more data)