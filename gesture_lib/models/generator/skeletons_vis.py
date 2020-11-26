import numpy as np
import matplotlib.pyplot as plt


TRT_TOPOLOGY = tuple([(1, 2, 17), (2, 3), (4, ), (5, ), (6, ),
                      (7, 17), (8, 17), (9, ), (10, ), (),
                      (), (12, 13, 17), (14, 17), (15, ),
                      (16, ), (), (), ()])

TOPOLOGY = {'trtpose': TRT_TOPOLOGY}


def visSkeleton3D(ax, points, topo="trtpose", xlim=[-1, 1], ylim=[-1, 1], zlim=[0, 3]):
    try:
        topology = TOPOLOGY[topo]
    except KeyError:
        print("invalid topology name!")
        raise
    if points.shape[0] == 0:
        return
    assert len(points.shape) == 3
    assert points.shape[2] == 3, "invalid point shape for visualization"
    for i in range(points.shape[0]):
        point = points[i]
        for idx, tps in enumerate(topology):
            if (point[idx, 2] < 1e-5) or (len(tps) == 0):
                continue
            for tp in tps:
                if point[tp, 2] < 1e-5:
                    continue
                x = [point[idx, 0], point[tp, 0]]
                y = [point[idx, 1], point[tp, 1]]
                z = [point[idx, 2], point[tp, 2]]
                ax.plot(x, y, z, c='g')
        point = point[point[:, -1] > 0]
        ax.scatter(point[:, 0], point[:, 1], point[:, 2], label=str(i))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
