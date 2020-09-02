import argparse
import sys
sys.path.append(".")
from gesture_lib.apis import Inference


def parse_args():
    parser = argparse.ArgumentParser(description="camera inference")
    parser.add_argument('config',
                        help="yaml config file path")
    parser.add_argument('checkpoint',
                        help="path to gesture model '.pth' checkpoint")
    parser.add_argument('--show', action='store_true',
                        help="show the recognition result or not")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cfg_path = args.config
    ckp = args.checkpoint
    if cfg_path.split('.')[-1] != 'yaml':
        print("invalid .yaml configure file!")
        exit(0)
    if ckp.split('.')[-1] != 'pth':
        print("invalid .pth checkpoint file!")
        exit(0)

    camera_infer = Inference(cfg_path=cfg_path, checkpoints=ckp)
    camera_infer.run(show=args.show)


if __name__ == "__main__":
    main()
