import os

import numpy as np
import matplotlib.pyplot as plt

from matplotlib import cm
from tslearn.barycenters import softdtw_barycenter
from tslearn.metrics.dtw_variants import dtw_path

def interpolateH2L(trajectory,scale):
    length = trajectory.shape[0]
    new_trajectory = np.zeros((int(length/scale),trajectory.shape[1]))
    for i in range(int(length/scale)):
        new_trajectory[i,:] = trajectory[i*scale,:]
    return new_trajectory

def quat_log(ori):
    quat_dist = np.zeros_like(ori)
    for i in range(len(ori)):
        temp_v = np.arccos(ori[i, :, 3])
        temp_u = 1 / np.linalg.norm(ori[i, :, :3], axis=1)
        for j in range(len(ori[i])):
            quat_dist[i, j, :3] = 2 * temp_v[j] * temp_u[j] * ori[i, j, :3]

    return quat_dist[:, :, :3]

def quat_exp(ori):
    #TODO
    quat = np.zeros((length, 4))

def plot_helper(X, barycenter, axe):
    t_index = np.linspace(0, 10, length)
    # plot all points of the data set
    for series in X:
        axe.plot(t_index, series.ravel(), "k-", alpha=.2)
    # plot the given barycenter of them
    axe.plot(t_index, barycenter.ravel(), "r-", linewidth=2)
    axe.label_outer()

def soft_dtw_plot(test_list, gamma, name):
    # Imp_mat
    centers = np.zeros((length, 3))
    # Soft-DTW align demo and plot
    fig, axes = plt.subplots(3)
    fig.suptitle("Soft-DTW {}  ($\gamma$={})".format(name, gamma))
    for i in range(3):
        X = test_list[:, :, i]
        soft_dba_centers = softdtw_barycenter(X, gamma=gamma, max_iter=50, tol=1e-3)
        centers[:, i] = soft_dba_centers.ravel()
        plot_helper(X, soft_dba_centers, axes[i])
    return centers

def warp_trajectories(center,dataset):
    aligned_trajectories = np.zeros_like(dataset)
    for s in range(n_series):
        path, dist = dtw_path(center,dataset[s, :])
        for id in range(length):
            aligned_trajectories[s, id] = dataset[s, path[id][1]]
    return aligned_trajectories

def stdev2imp(stdev,min_imp,max_imp):
    """
    :param stdev: n_length X 3
    :param min_imp: constant
    :param max_imp: constant
    :return: n_length X 3
    """
    min_std = np.min(stdev)
    max_std = np.max(stdev)
    # in a(x-max_var)^2+min_impedance
    a = (max_imp - min_imp) / (min_std - max_std)**2
    imp_mat = a * (stdev - max_std)**2 + min_imp
    return imp_mat

if __name__ == '__main__':
    # 0) fetch the example data set
    np.random.seed(0)
    demos = np.load(os.path.join(os.path.curdir,'./data/pouring_water.npy'))
    n_series = len(demos)
    length = 200

    # 1) Interpolate demonstrated trajectory
    test_pose_list = np.zeros((n_series, length, 7))
    for id in range(n_series):
        test_pose_list[id, :] = interpolateH2L(demos[id, :],scale=int(len(demos[0])/length))
    # 2) Soft-DTW encode position trajectory and plot
    test_pos_list = test_pose_list[:, :, :3]
    test_ori_list = quat_log(test_pose_list[:, :, 3:])
    pos_centers = soft_dtw_plot(test_pos_list, gamma=0.1, name="Position")
    ori_centers = soft_dtw_plot(test_ori_list, gamma=0.1, name="Orientation")

    # 2.1) warp trajectories
    warped_pos_traj = np.zeros_like(test_pos_list)
    warped_ori_traj = np.zeros_like(test_ori_list)
    for id in range(3):
        warped_pos_traj[:, :, id] = warp_trajectories(pos_centers[:, id], test_pos_list[:, :, id])
        warped_ori_traj[:, :, id] = warp_trajectories(ori_centers[:, id], test_ori_list[:, :, id])
        # Use euclidean centers of aligned trajectories
        pos_centers[:, id] = np.mean(warped_pos_traj[:, :, id], axis=0)
        ori_centers[:, id] = np.mean(warped_ori_traj[:, :, id], axis=0)

    # 3) Calculate standard deviation
    pos_std_mat = np.std(warped_pos_traj, axis=0)
    ori_std_mat = np.std(warped_ori_traj, axis=0)

    # 4) Calculate impedance
    pos_imp_mat = stdev2imp(pos_std_mat, 200, 500)
    ori_imp_mat = stdev2imp(ori_std_mat, 10, 20)

    # 5) Plot
    t_index = np.linspace(0, 10, length)
    fig, axes = plt.subplots(3)
    plt.suptitle('DTW Aligned Position')
    for id in range(3):
        for s in range(n_series):
            axes[id].plot(t_index, warped_pos_traj[s, :, id], color=cm.viridis(0.7), alpha=0.3)
        axes[id].plot(t_index, pos_centers[:, id], color=cm.viridis(0.3))
        axes[id].fill_between(t_index, pos_centers[:, id] - pos_std_mat[:, id],
                             pos_centers[:, id] + pos_std_mat[:, id], color=cm.viridis(0.3), alpha=0.4)
        axes[id].label_outer()

    fig, axes = plt.subplots(3)
    plt.suptitle('DTW Aligned Orientation')
    for id in range(3):
        for s in range(n_series):
            axes[id].plot(t_index, warped_ori_traj[s, :, id], color=cm.viridis(0.7), alpha=0.3)
        axes[id].plot(t_index, ori_centers[:, id], color=cm.viridis(0.3))
        axes[id].fill_between(t_index, ori_centers[:, id] - ori_std_mat[:, id],
                             ori_centers[:, id] + ori_std_mat[:, id], color=cm.viridis(0.3), alpha=0.4)
        axes[id].label_outer()

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.plot(t_index,pos_imp_mat[:, 0], label="pos_x")
    ax1.plot(t_index,pos_imp_mat[:, 1], label="pos_y")
    ax1.plot(t_index,pos_imp_mat[:, 2], label="pos_z")
    ax1.set_title("Translational Stiffness Profiles")
    ax1.label_outer()
    ax1.legend()

    ax2.plot(t_index, ori_imp_mat[:, 0], label="ori_x")
    ax2.plot(t_index, ori_imp_mat[:, 1], label="ori_y")
    ax2.plot(t_index, ori_imp_mat[:, 2], label="ori_z")
    ax2.set_title("Rotational Stiffness Profiles")
    ax2.label_outer()
    ax2.legend()
    plt.show()

    # 6) Save files
