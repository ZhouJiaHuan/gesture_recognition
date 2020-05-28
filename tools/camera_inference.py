import argparse
import sys
sys.path.append(".")
from gesture_lib.apis import InferenceSurf, InferenceDlib


def parse_args():
    parser = argparse.ArgumentParser(description="camera inference")
    parser.add_argument('config',
                        help="yaml config file path")
    parser.add_argument('checkpoint',
                        help="path to gesture model '.pth' checkpoint")
    parser.add_argument('--op_model',
                        help="path to openpose model folder, ignored when using trt-pose")
    parser.add_argument('--trt_model',
                        help="path to trt model, ignored when using openpose")
    parser.add_argument('--trt_json',
                        help="path to trt pose json file, ignored when using openpose")
    parser.add_argument('--seq_len', type=int, default=30,
                        help="sequence length for recognition, default 30")
    parser.add_argument('--dlib', action='store_true',
                        help="whether to use dlib for face feature extraction or not")
    parser.add_argument('--show', action='store_true',
                        help="show the recognition result or not")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_path = args.config
    ckp = args.checkpoint
    if cfg_path is None or ckp is None:
        print("configure path and checkpoint must be specified!")
        raise
    if cfg_path.split('.')[-1] != 'yaml':
        print("invalid .yaml configure file!")
        raise
    if ckp.split('.')[-1] != 'pth':
        print("invalid .pth checkpoint file!")
        raise

    if args.dlib:
        camera_infer = InferenceDlib(cfg_path=cfg_path,
                                     checkpoints=ckp,
                                     seq_len=args.seq_len,
                                     op_model=args.op_model,
                                     trt_model=args.trt_model,
                                     pose_json_path=args.trt_json,
                                     )
    else:
        camera_infer = InferenceSurf(cfg_path=cfg_path,
                                     checkpoints=ckp,
                                     seq_len=args.seq_len,
                                     op_model=args.op_model,
                                     trt_model=args.trt_model,
                                     pose_json_path=args.trt_json,
                                     )
    camera_infer.infer_pipeline(show=args.show)


if __name__ == "__main__":
    main()
