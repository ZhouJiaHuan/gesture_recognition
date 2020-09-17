import sys
sys.path.append('.')
import time
from multiprocessing import Queue, Process

from gesture_lib.apis.inference_fisheye_process import producer, consumer


if __name__ == "__main__":
    q1 = Queue(maxsize=5)
    p1 = Process(target=producer, args=(q1, ))
    c1 = Process(target=consumer, args=(q1, True))

    p1.start()
    time.sleep(1)
    c1.start()
    p1.join()
    c1.join()
