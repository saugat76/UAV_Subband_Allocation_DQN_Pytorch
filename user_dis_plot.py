import gym
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import random

 ## Polar to Cartesian and vice versa
def pol2cart(r,theta):
    return (r * np.cos(theta), r * np.sin(theta))

def cart2pol(z):
        return (np.abs(z), np.angle(z))


NUM_USER = 100
NUM_UAV = 5
COVERAGE_XY = 1000
SEED = 1
random.seed(SEED)

HOTSPOTS = np.array(
        [[200, 200], [800, 800], [300, 800], [800, 300]])  # Position setup in grid size rather than actual distance
USER_DIS = int(NUM_USER / NUM_UAV)
USER_LOC = np.zeros((NUM_USER - USER_DIS, 3))

for i in range(len(HOTSPOTS)):
    for j in range(USER_DIS):
        temp_loc_r = random.uniform(-(1/3.5)*COVERAGE_XY, (1/3.5)*COVERAGE_XY)
        temp_loc_theta = random.uniform(0, 2*math.pi)
        temp_loc = pol2cart(temp_loc_r, temp_loc_theta)
        (temp_loc_1, temp_loc_2) = temp_loc
        temp_loc_1 = temp_loc_1 + HOTSPOTS[i, 0]
        temp_loc_2 = temp_loc_2 + HOTSPOTS[i, 1]
        USER_LOC[i * USER_DIS + j, :] = [temp_loc_1, temp_loc_2, i]
temp_loc = np.random.uniform(low=0, high=COVERAGE_XY, size=(USER_DIS, 3))
temp_loc[:, 2] = 5
USER_LOC = np.concatenate((USER_LOC, temp_loc))
print(USER_LOC)

# USER_LOC[USER_LOC[:,2] == 0, 2] = '#ff0000'
# USER_LOC[USER_LOC[:,2] == 0, 2] = '#003153'
# USER_LOC[USER_LOC[:,2] == 0, 2] = '#197f7f'
# USER_LOC[USER_LOC[:,2] == 0, 2] = '#ffba00'
# USER_LOC[USER_LOC[:,2] == 0, 2] = '#99ff66'

position = HOTSPOTS
fig = plt.figure(figsize=(6, 6), dpi=200)
gs = GridSpec(1, 1, figure=fig)
ax = fig.add_subplot(gs[0:1, 0:1])
ax.cla()
for i in range(0, 100):
    if USER_LOC[i,2] == 0:
        co = '#CB4335'
    if USER_LOC[i,2] == 1:
        co = '#7D3C98'
    if USER_LOC[i,2] == 2:
        co = '#2E86C1'
    if USER_LOC[i,2] == 3:
        co = '#28B463'
    if USER_LOC[i,2] == 5:
        co = '#F1C40F'

    ax.scatter(USER_LOC[i, 0], USER_LOC[i, 1], c = co, marker='o')
ax.scatter(position[:, 0], position[:, 1], c = '#000000', marker='x', label = 'Hotspot center')
ax.scatter([], [], c = '#000000', marker='o', label = "User" )

# for (i,j) in (position[:,:]):
#     cc = plt.Circle((i,j), (1/3.5)*1000, alpha=0.1)
#     ax.set_aspect(1)
#     ax.add_artist(cc)

# plt.pause(0.5)
plt.legend()
plt.xlim(-50, 1050)
plt.ylim(-50, 1050)
plt.draw()
plt.show()


