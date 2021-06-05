import numpy as np
import matplotlib.pyplot as plt

from tslearn.barycenters import euclidean_barycenter, softdtw_barycenter
from tslearn.metrics.dtw_variants import dtw_path
from tslearn.datasets import CachedDatasets

# fetch the example data set
np.random.seed(0)
X_train, y_train, _, _ = CachedDatasets().load_dataset("Trace")
X = X_train[y_train == 2]
length_of_sequence = X.shape[1]

def warp_trajectories(center,dataset):
    aligned_trajectories = np.zeros_like(np.asarray(dataset))
    for s in range(len(dataset)):
        path, dist = dtw_path(center,dataset[s, :])
        for id in range(length_of_sequence):
            aligned_trajectories[s, id] = dataset[s, path[id][1]]
    return aligned_trajectories

def plot_helper(barycenter):
    # plot all points of the data set
    for series in X:
        plt.plot(series.ravel(), "k-", alpha=.2)
    # plot the given barycenter of them
    plt.plot(barycenter.ravel(), "r-", linewidth=2)


# plot the four variants with the same number of iterations and a tolerance of
# 1e-3 where applicable
ax1 = plt.subplot(3, 1, 1)
plt.title("Euclidean barycenter")
eul_centers = euclidean_barycenter(X)
plot_helper(eul_centers)

plt.subplot(3, 1, 2, sharex=ax1)
plt.title("Soft-DTW barycenter ($\gamma$=1.0)")
sdtw_centers = softdtw_barycenter(X, gamma=1., max_iter=50, tol=1e-3)
plot_helper(sdtw_centers)

plt.subplot(3, 1, 3, sharex=ax1)
plt.title("Soft-DTW2 Warped Trajectories")
aligned_trajectories = warp_trajectories(sdtw_centers, X)
for s in range(len(aligned_trajectories)):
    plt.plot(aligned_trajectories[s], "k-", alpha=.2)
plt.plot(euclidean_barycenter(aligned_trajectories))
# clip the axes for better readability
ax1.set_xlim([0, length_of_sequence])

# show the plot(s)
plt.tight_layout()
plt.show()