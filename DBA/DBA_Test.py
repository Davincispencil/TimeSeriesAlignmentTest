import os
import numpy as np

import matplotlib.pyplot as plt


from .DBA import performDBA as simple_dba
from .DBA_multivariate import performDBA as dba

def interpolateH2L(trajectory,scale):
    length = trajectory.shape[0]
    new_trajectory = np.zeros((int(length/scale),trajectory.shape[1]))
    for i in range(int(length/scale)):
        new_trajectory[i,:] = trajectory[i*scale,:]
    return new_trajectory


if __name__ == '__main__':
    demos = np.load(os.path.join(os.path.curdir,'../data/pouring_water.npy'))
    n_series = len(demos)
    length = 100

    # 0) Plot each dimension of demos and save
    def plot_demos(demos,length,dim,n_series,name):
        t_index = np.linspace(0,10,length)
        fig, axes = plt.subplots(dim)
        for id in range(n_series):
            axes[0].plot(t_index, demos[id, :, 0])
            axes[1].plot(t_index, demos[id, :, 1])
            axes[2].plot(t_index, demos[id, :, 2])

        for ax in fig.get_axes():
            ax.label_outer()

        plt.savefig("plot/{}.pdf".format(name))

    # 1) Interpolate demo
    demo_tra_list = demos[:, :, :3]
    test_tra_list = np.zeros((n_series, length, 3))
    for id in range(n_series):
        test_tra_list[id, :] = interpolateH2L(demo_tra_list[id, :],
                                           scale=int(len(demo_tra_list[id, :])/length))
    # plot_demos(test_tra_list,length,3,n_series,name="Pos4DBA")

    # 2) Test 1D Alignment using demos
    fig, axes = plt.subplots(3)
    t_index = np.linspace(0, 10, length)
    for id in range(3):
        series = list(test_tra_list[:, :, id])
        for s in series:
            axes[id].scatter(t_index, s, s=2)

        average_series = simple_dba(series)

        axes[id].plot(t_index, average_series, linewidth=3, color='r')
    plt.show()
    # 3.1) Test 3D Alignment (Positions)

    # 3.2) Test 3D Alignment (Orientations)


