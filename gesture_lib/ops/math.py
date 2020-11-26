import math


def cross_point(point11, point12, point21, point22, img_size):
    '''compute cross point with 2 point

    Args:
        point11, point12: [list], [x, y]
        point21, point22: [list], [x, y]
        img_size: [list], [w, h], img size

    Return:
        point: [list], [x, y], if no valid cross point found, x == y == -1
    '''
    point = [-1, -1]
    w, h = img_size
    x11, y11 = point11
    x12, y12 = point12
    x21, y21 = point21
    x22, y22 = point22

    # case 1: 2 vertical lines
    if x11 == x12 and x21 == x22:
        return point

    # case 2: 1 vertical line
    if x11 == x12 and x21 != x22:
        k2 = (y22-y21) / (x22-x21)
        b2 = y21 - k2*x21
        x = x11
        y = k2 * x + b2
        point = [x, y]
        return point

    if x11 != x12 and x21 == x22:
        k1 = (y12-y11) / (x12-x11)
        b1 = y11 - k1*x11
        x = x21
        y = k1 * x + b1
        point = [x, y]
        return point

    k1 = (y12-y11) / (x12-x11)
    k2 = (y22-y21) / (x22-x21)

    # case 3: 2 parrallel lines
    if k1 == k2:
        return point

    # case 4
    b1 = y11 - k1*x11
    b2 = y21 - k2*x21

    x = (b2-b1) / (k1-k2)
    y = x*k1 + b1

    if x < 0 or x > w or y < 0 or y > h:
        return point
    point = [x, y]
    return point


def cross_point_in(point11, point12, point21, point22, img_size):
    '''compute inline cross point
    '''
    point = cross_point(point11, point12, point21, point22, img_size)
    if point[0] < min(point11[0], point12[0]):
        return [-1, -1]
    if point[0] > max(point11[0], point12[0]):
        return [-1, -1]
    if point[1] < min(point11[1], point12[1]):
        return [-1, -1]
    if point[1] > max(point11[1], point12[1]):
        return [-1, -1]

    if point[0] < min(point21[0], point22[0]):
        return [-1, -1]
    if point[0] > max(point21[0], point22[0]):
        return [-1, -1]
    if point[1] < min(point21[1], point22[1]):
        return [-1, -1]
    if point[1] > max(point21[1], point22[1]):
        return [-1, -1]

    return point


def computeP3withD(point1, point2, d):
    '''
    d is the distance between target point and point1
    '''
    dim = len(point1)
    if dim == 2:
        x1, y1 = point1
        x2, y2 = point2
        k = d / math.sqrt((x2-x1)**2 + (y2-y1)**2)
        x = k * (x2 - x1) + x1
        y = k * (y2 - y1) + y1
        return (x, y)
    elif dim == 3:
        x1, y1, z1 = point1
        x2, y2, z2 = point2
        k = d / math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)
        x = k * (x2 - x1) + x1
        y = k * (y2 - y1) + y1
        z = k * (z2 - z1) + z1
        return (x, y, z)
    else:
        raise KeyError


def test01():
    point11 = [320, 0]
    point12 = [320, 480]
    point21 = [0, 240]
    point22 = [640, 240]
    img_size = [640, 480]
    point = cross_point_in(point11, point12, point21, point22, img_size)

    print(point)


def test02():
    point11 = (0, 0, 0)
    point12 = (10, 10, 10)
    d = 20

    point = computeP3withD(point11, point12, d)
    print(point)


if __name__ == '__main__':
    test02()
