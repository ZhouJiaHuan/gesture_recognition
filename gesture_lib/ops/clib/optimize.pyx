import numpy as np
cimport numpy as np

DTYPE = np.uint8
ctypedef np.uint8_t DTYPE_t

def dist(np.ndarray[DTYPE_t, ndim=2] kernel,
         np.ndarray[DTYPE_t, ndim=2] img,
         int x,
         int y,
         int k_w,
         int k_h):
    cdef float d
    cdef np.ndarray[DTYPE_t, ndim=2] region
    region = img[y-k_h//2:y+k_h//2+1, x-k_w//2:x+k_w//2+1]
    d = np.sqrt(np.sum((region-kernel)**2))
    return d


def local_search(np.ndarray[DTYPE_t, ndim=2] kernel,
                 np.ndarray[DTYPE_t, ndim=2] rgbd_img,
                 np.ndarray kp,
                 tuple search_size=(25, 25)):
    ''' keypoint local search
    Args:
        kernel: a small kernel cropped from fish image
        rgbd_img: gray image from rgbd camera
        kp: current keypoint on rgbd image mapped from fisheye image
        search_size: search area size on rgbd image

    Return:
        result_kp: optimized keypoint on rgbd image
    '''
    cdef int k_h, k_w, s_h, s_w, img_h, img_w
    k_h, k_w = kernel.shape[:2]
    s_h, s_w = search_size
    img_h, img_w = rgbd_img.shape[:2]

    # case 0: kernel size is larger than search size
    if k_h >= s_h or k_w >= s_w:
        return kp
    # case 1: source keypoint is out of the rgbd image
    if kp[0] < 0 or kp[0] > img_w-1 or kp[1] < 0 or kp[1] > img_h-1:
        return kp
    cdef int x1, y1, x2, y2

    x1 = max(k_w // 2, int(kp[0] - s_w // 2 + k_w // 2))
    y1 = max(k_h // 2, int(kp[1] - s_h // 2 + k_h // 2))
    x2 = min(img_w-k_w // 2, int(kp[0] + s_w // 2 - k_w // 2))
    y2 = min(img_h-k_h // 2, int(kp[1] + s_h // 2 - k_h // 2))

    # case 2: invalid search region
    if x1 >= x2 or y1 >= y2:
        return kp

    # case 3:
    cdef int x, y, idx
    xx = list(range(x1, x2)) * (y2-y1+1)
    yy = list(range(y1, y2)) * (x2-x1+1)
    dists = []
    for x, y  in zip(xx, yy):
        dists.append(dist(kernel, rgbd_img, x, y, k_w, k_h))
    idx = dists.index(min(dists))
    result_kp = [xx[idx], yy[idx]]
    return result_kp
