import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import numpy as np

# Plot the grid space
fig = plt.figure()
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0:1, 0:1])
THETA = 60 * math.pi / 180
grid_space = 100
coverage_radius = 350 * np.tan(60 * math.pi / 180 / 2)
u_loc = np.loadtxt('UserLocation.txt', delimiter=' ').astype(np.int64)
state = np.zeros((5, 3), dtype=np.int32)
state[:, 0:2] = [[3, 0], [3, 4], [2, 9], [8, 7], [9, 2]]


ax.cla()
position = state[:, 0:2] * grid_space
ax.scatter(u_loc[:, 0], u_loc[:, 1], c = '#ff0000', marker='o', label = "Users")
ax.scatter(position[:, 0], position[:, 1], c = '#000000', marker='x', label = "UAV")
for (i,j) in (position[:,:]):
    cc = plt.Circle((i,j), coverage_radius, alpha=0.1)
    ax.set_aspect(1)
    ax.add_artist(cc)
ax.legend(loc="lower right")
plt.show()