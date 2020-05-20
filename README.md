This repository realize the simple gestures recognition based on skeletons information extracted by Openpose or Trt-pose. The main process includes 5 key steps

- get a sequence of frames from the depth camera (eg., realsense), including the color frame and depth frame
- extract the skeletons informations with `openpose body-25` model or `trt-pose body-18` model.
- track person with  `SURF` or `AKAZE` feature
- parse the 3-d location of the 25 body key points from realsense camera
- predict the gestures from fixed length of frames (eg., 30 frames) with LSTM model

# Dependence

- python 3.6
- OpenCV 3.4.2 (with extra modules)
- PyTorch 1.4 (with torchvision)
- openpose library
- realsense library
- TensorRT: https://github.com/NVIDIA/TensorRT
- torch2trt: https://github.com/NVIDIA-AI-IOT/torch2trt
- trt-pose: https://github.com/NVIDIA-AI-IOT/trt_pose

The `TensorRT` and `torch2trt` are used for converting and loading `trt-pose` model.

# Dataset

The dataset for training and test includes 4 simple gestures, namely, `wave`, `come`, `hello` and `others`. The source samples are collected by realsense and saved as `.bag` files which consist of the color frame and depth frame information.

For LSTM training, body key points information are extracted from the color image in bag file and the 3-D locations of the 25 key points are computed with realsense api. Eventually, each sample for training and test is represented by a T * 100 or T * 75 matrix saved in a txt file, where T is the number of frames related to a gesture and 100 = 25 * (3+1) , namely the x, y, z (meters) with scores of the 25 key points for openpose model. For trt-pose, 75 = 25 * 3, namely the x, y, z without scores.

# Model

We use the `openpose body-25` model (**higher accuracy**) and `trt-pose body-18` model (**higher speed**) for human pose estimation.

The model for gesture recognition is a `LSTM` model,  which generally includes 2 LSTM layers and 1 fully connected layer with the SoftMax for classification, see `models/lstm.py` for more details.

# Train

For training the model, just run:

```bash
python tools/train.py CONFIGURE_FILE
```

the configure file is specified by `.yaml` file, see example in `configs/lstm_op-body25.yaml` and `configs/lstm_trt-body18.yaml`  for more details.

# Inference

For the inference with realsense camera, run:

```
python tools/camera_inference.py CONFIG CHECKPOINT --op_model OP_MODEL] 
       --trt_model TRT_MODEL --trt_json TRT_JSON --extractor EXTRACTOR 
       --seq_len SEQ_LEN --show                    
```

For specified example, see `camera_inference_op.sh` and `camera_inference_trt.sh`.
