import argparse
import sys
import time
from collections import deque
sys.path.append(".")
from gesture_lib.apis.inference_fisheye_thread import FisheyeStream, PoseEst


def parse_args():
    parser = argparse.ArgumentParser(description="camera inference")
    parser.add_argument('--show', action='store_true',
                        help="show the recognition result or not")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    cache = deque([], 5)
    fisheye = FisheyeStream(cache)
    pose_est = PoseEst(cache, show=args.show)
    fisheye.start()
    time.sleep(1)
    pose_est.start()

    fisheye.join()
    pose_est.join()


if __name__ == "__main__":
    main()
