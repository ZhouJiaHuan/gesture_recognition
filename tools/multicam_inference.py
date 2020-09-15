import argparse
import sys
sys.path.append('.')
from gesture_lib.apis import InferenceMulticam


def parse_args():
    parser = argparse.ArgumentParser(description="camera inference")
    parser.add_argument('--show', action='store_true',
                        help="show the recognition result or not")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    camera_infer = InferenceMulticam()
    camera_infer.run(show=args.show)


if __name__ == "__main__":
    main()

