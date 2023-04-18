from matplotlib import gridspec
import matplotlib.pyplot as plt
from uav_env import UAVenv
import numpy as np 
from matplotlib.gridspec import GridSpec
import math

def final_render(state, remark):
    if remark == "best_user1":
        USER_LOC = np.loadtxt('UserLocation_1.txt', delimiter=' ').astype(np.int64)
    elif remark == "best_user2":
        USER_LOC = np.loadtxt('UserLocation_2.txt', delimiter=' ').astype(np.int64)
    u_loc = USER_LOC
    fig = plt.figure()
    gs = GridSpec(1, 1, figure=fig)
    ax = fig.add_subplot(gs[0:1, 0:1])
    grid_space = 100
    UAV_HEIGHT = 350
    THETA = 60 * math.pi / 180
    coverage_radius = UAV_HEIGHT * np.tan(THETA / 2)
    ax.cla()
    position = state[:, 0:2] * grid_space
    ax.scatter(u_loc[:, 0], u_loc[:, 1], c = '#ff0000', marker='o', label = "Users")
    ax.scatter(position[:, 0], position[:, 1], c = '#000000', marker='x', label = "UAV")
    for (i,j) in (position[:,:]):
        cc = plt.Circle((i,j), coverage_radius, alpha=0.1)
        ax.set_aspect(1)
        ax.add_artist(cc)
    ax.legend()
    if remark == "best_user1":
        plt.title("Best state of UAV: Before Change")
    elif remark == "best_user2":
        plt.title("Best state of UAV: After Change")
    plt.pause(0.5)
    plt.xlim(-50, 1050)
    plt.ylim(-50, 1050)