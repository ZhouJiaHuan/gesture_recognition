def box_iou(box1, box2, mode='iou'):
    assert mode in ['iou', 'diou']
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    area1 = (x12-x11+1) * (y12-y11+1)
    area2 = (x22-x21+1) * (y22-y21+1)

    xx1, xx2 = max(x11, x21), min(x12, x22)
    yy1, yy2 = max(y11, y21), min(y12, y22)
    inter = max(0, xx2-xx1+1) * max(0, yy2-yy1+1)

    iou = inter / (area1 + area2 - inter)

    if mode == 'iou':
        return iou
    elif mode == 'diou':
        c1_x, c1_y = (x11 + x12) / 2, (y11 + y12) / 2
        c2_x, c2_y = (x21 + x22) / 2, (y21 + y22) / 2
        d = (c1_x-c2_x) ** 2 + (c1_y-c2_y) ** 2
        xx1, xx2 = min(x11, x21), max(x12, x22)
        yy1, yy2 = min(y11, y21), max(y12, y22)
        c = (xx2-xx1) ** 2 + (yy2-yy1) ** 2
        diou = iou - d / c
        return diou
    else:
        raise
