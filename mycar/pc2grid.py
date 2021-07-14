import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection


def pc2grid():
    pc = np.genfromtxt('depth0.csv', delimiter=',')
    # remove points outside of area of interest and empty points
    pc = pc[np.linalg.norm(pc, axis=1) < 3]
    pc = pc[pc[:, 0] != 0]
    # resize area of interest in y axis
    pc = pc[pc[:, 1] < 0.23]
    pc = pc[pc[:, 1] > 0.15]
    # remove duplicate points in relation to resolution
    # tol = 0.001
    # pc = pc[~(np.triu(np.abs(pc[:, 0] - pc[:, 0]) <= tol, 1)).any(0)]
    # remove points with the same x and different z
    pc = pc.round(3)
    pc = pc[np.unique(pc[:, 2], axis=0, return_index=True)[1]]
    forplot = np.array([pc[:, 0], pc[:, 2]])
    print(forplot)
    forplot_1 = np.floor(forplot*10)/10
    forplot_2 = abs(forplot_1 - np.ceil(forplot*10)/10)
    # plt.pcolormesh(forplot)
    plt.xlim((-2, 2))
    plt.ylim((-1, 3))
    xticks = np.arange(-2, 2.1, 0.1)
    yticks = np.arange(-1, 3.1, 0.1)
    plt.plot(forplot[0], forplot[1], 'o')
    plt.show()
    # rect = []
    # for i in range(0, forplot.shape[1]):
    #     rect.append(patches.Rectangle((forplot_1[0, i], forplot_1[1, i]), forplot_2[0, i], forplot_2[1, i],linewidth=1, edgecolor='none', facecolor='blue'))
    # fig = plt.figure()
    # ax = fig.add_subplot(111, aspect='equal')
    # ax.set_xlim(-2, 2)
    # ax.set_ylim((-1, 3))
    # xticks = np.arange(-2, 2.1, 0.1)
    # yticks = np.arange(-1, 3.1, 0.1)
    # ax.set_xticks(xticks)
    # ax.set_yticks(yticks)
    # ax.grid()
    # ax.add_collection(PatchCollection(rect))
    # # ax = fig.add_subplot(projection='3d')
    # # ax.scatter(pc[:, 0], pc[:, 1], pc[:, 2])
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # # ax.set_zlabel('z')
    # plt.show()


# def matlab_():
#     import matlab.engine
#     eng = matlab.eng


if __name__ == '__main__':
    pc2grid()
    # matlab_()
