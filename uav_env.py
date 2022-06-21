
###################################
## Environment Setup of for UAV  ##
###################################

from turtle import pos
import gym
from gym import spaces
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random


###################################
##     UAV Class Defination      ##
###################################

class UAVenv(gym.Env):
    """Custom Environment that follows gym interface """
    metadata = {'render.modes': ['human']}

    # Fixed Input Parameters
    NUM_USER = 100  # Number of ground user
    NUM_UAV = 5  # Number of UAV
    Fc = 2  # Operating Frequency 2 GHz
    LightSpeed = 3 * (10 ** 8)  # Speed of Light
    WaveLength = LightSpeed / (Fc * (10 ** 9))  # Wavelength of the wave
    COVERAGE_XY = 1000
    UAV_HEIGHT = 350
    BS_LOC = np.zeros((NUM_UAV, 3))
    THETA = 60 * math.pi / 180  # in radian   # Bandwidth for a resource block (This value is representing 2*theta instead of theta)
    BW_UAV = 4e6  # Total Bandwidth per UAV   # Update to decrease the available BW
    BW_RB = 180e3  # Bandwidth of a Resource Block
    ACTUAL_BW_UAV = BW_UAV * 0.9
    grid_space = 100
    GRID_SIZE = int(COVERAGE_XY / grid_space)  # Each grid defined as 100m block

    
    ## Polar to Cartesian and vice versa
    def pol2cart(r,theta):
        return (r * np.cos(theta), r * np.sin(theta))
    
    def cart2pol(z):
        return (np.abs(z), np.angle(z))

    # User distribution on the target area // NUM_USER/5 users in each of four hotspots
    # Remaining NUM_USER/5 is then uniformly distributed in the target area

    HOTSPOTS = np.array(
        [[200, 200], [800, 800], [300, 800], [800, 300]])  # Position setup in grid size rather than actual distance
    USER_DIS = int(NUM_USER / NUM_UAV)
    USER_LOC = np.zeros((NUM_USER - USER_DIS, 2))
    
    for i in range(len(HOTSPOTS)):
        for j in range(USER_DIS):
            temp_loc_r = random.uniform(-(1/3.5)*COVERAGE_XY, (1/3.5)*COVERAGE_XY)
            temp_loc_theta = random.uniform(0, 2*math.pi)
            temp_loc = pol2cart(temp_loc_r, temp_loc_theta)
            (temp_loc_1, temp_loc_2) = temp_loc
            temp_loc_1 = temp_loc_1 + HOTSPOTS[i, 0]
            temp_loc_2 = temp_loc_2 + HOTSPOTS[i, 1]
            USER_LOC[i * USER_DIS + j, :] = [temp_loc_1, temp_loc_2]
    temp_loc = np.random.uniform(low=0, high=COVERAGE_XY, size=(USER_DIS, 2))
    USER_LOC = np.concatenate((USER_LOC, temp_loc))
    np.savetxt('UserLocation.txt', USER_LOC, fmt='%.3e', delimiter=' ', newline='\n')

    # Saving the user location on a file instead of generating everytime

    USER_LOC = np.loadtxt('UserLocation.txt', dtype=np.int32, delimiter=' ')

    def __init__(self):
        super(UAVenv, self).__init__()
        # Defining action spaces // UAV RB allocation to each user increase each by 1 until remains
        # Five different action for the movement of each UAV
        # 1 = Right, 2 = Left, 3 = straight, 4 = back ,5 = Hover
        self.action_space = spaces.Discrete(5)
        # Defining Observation spaces // UAV RB to each user
        # Position of the UAV in space // constant height and X and Y pos
        # self.observation_space = spaces.Discrete(self.NUM_UAV)
        self.u_loc = self.USER_LOC
        self.state = np.zeros((self.NUM_UAV, 3), dtype=np.int32)
        # set the states to the hotspots and one at the centre for faster convergence
        # further complexity by choosing random value of state
        # self.state[:, 0:2] = [[1, 2], [4, 2], [7, 3], [3, 8], [4, 5]]
        self.state[:, 0:2] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.state[:, 2] = self.UAV_HEIGHT
        self.coverage_radius = self.UAV_HEIGHT * np.tan(self.THETA / 2)
        print(self.coverage_radius)

    def step(self, action):
        # Assignment of sub carrier band to users
        # Reshape of actions
        # Execution of one step within the environment
        # Deal with out of boundaries conditions
        isDone = False
        flag = 0
        previous_reward = 0
        # Calculate the distance of every users to the UAV BS and organize as a list
        dist_u_uav = np.zeros(shape=(self.NUM_UAV, self.NUM_USER))
        for i in range(self.NUM_UAV):
            tem_x = self.state[i, 0]
            tem_y = self.state[i, 1]
            # one step action
            if action[i] == 1:
                self.state[i, 0] = self.state[i, 0] + 1
            elif action[i] == 2:
                self.state[i, 0] = self.state[i, 0] - 1
            elif action[i] == 3:
                self.state[i, 1] = self.state[i, 1] + 1
            elif action[i] == 4:
                self.state[i, 1] = self.state[i, 1] - 1
            elif action[i] == 5:
                pass
            else:
                print("Error Action Value")

            # Take boundary condition into account
            if self.state[i, 0] < 0 or self.state[i, 0] > self.GRID_SIZE or self.state[i, 1] < 0 or self.state[i, 1] > \
                    self.GRID_SIZE:
                self.state[i, 0] = tem_x
                self.state[i, 1] = tem_y
                flag += 1  # Later punish in reward function

        # Calculation of the distance value for all UAV and User
        for k in range(self.NUM_UAV):
            for l in range(self.NUM_USER):
                dist_u_uav[k, l] = math.sqrt((self.u_loc[l, 0] - (self.state[k, 0] * self.grid_space)) ** 2 + (self.u_loc[l, 1] -
                                                                                      (self.state[k, 1] * self.grid_space)) ** 2)
        max_user_num = self.ACTUAL_BW_UAV / self.BW_RB

        ######################
        ## Final Algorithm  ##
        ######################

        # User association to the UAV based on the distance value. First do a single sweep by all
        # the Users to request to connect to the closest UAV After the first sweep is complete the UAV will admit a
        # certain Number of Users based on available resource In the second sweep the User will request to the UAV
        # that is closest to it and UAV will admit the User if resource available

        # Connection request is a np array matrix that contains UAV Number as row and
        # User Number Connected to it on Columns and is stored in individual UAV to keep track of the
        # User requesting to connect

        connection_request = np.zeros(shape=(self.NUM_UAV, self.NUM_USER), dtype="int")

        for i in range(self.NUM_USER):
            if not(np.any(connection_request[:,i] == 1)):                 # Skip if connection request already sent
                close_uav = np.argmin(dist_u_uav[:,i])                    # Closest UAV index
                if dist_u_uav[close_uav, i] <= self.coverage_radius:      # UAV - User distance within the coverage radius then only connection request
                    connection_request[close_uav, i] = 1                  # All staifies, then connection request for the UAV - User


        # Allocating only 70% of max cap in first run
        # After all the user has send their connection request,
        # UAV only admits Users closest to and if bandwidth is available
        user_asso_flag = np.zeros(shape=(self.NUM_UAV, self.NUM_USER), dtype="int")
        for i in range(self.NUM_UAV):
            # Maximum Capacity for a single UAV
            cap_user_num = int(1 * max_user_num)
            # Sorting the users with the connection request to this UAV
            temp_user = np.where(connection_request[i, :] == 1)
            temp_user_distance = dist_u_uav[i, temp_user]
            temp_user_sorted = np.argsort(temp_user_distance) # Contains user index with closest 2D distance value (out of requested user)
            # The user list are already sorted, to associate flag bit of user upto the index from
            # min(max_user, max_number_of_user_inside_coverage_area)
            temp_user_idx = temp_user_sorted[0, 0:(min(cap_user_num, (np.size(temp_user_sorted)))-1)]
            # Index for the mid numpy array
            temp_user = np.array(temp_user)
            # Actual index of the users that send connection request, selected using distance value within the defined capacity
            temp_user_actual_idx = temp_user[0, temp_user_idx]
            # Set user association flag to 1 for that UAV and closest user index
            user_asso_flag[i, temp_user_actual_idx] = 1


        # For the second sweep, sweep through all users
        # If the user is not associated choose the closest UAV and check whether it has any available resource
        # If so allocate the resource and set the User association flag bit of that user to 1
        for j in range(self.NUM_USER):
            if not(np.any(user_asso_flag[:, j] != 0)):
                close_uav_id = dist_u_uav[:, j]
                close_uav_id = [i[0] for i in sorted(enumerate(close_uav_id), key=lambda x: x[1])]
                if dist_u_uav[close_uav_id[0], j] <= self.coverage_radius:
                    for close_id in close_uav_id:
                        if np.sum(user_asso_flag[close_id,:]) < max_user_num:
                            user_asso_flag[close_id, j] = 1
                            break


        # Need to work on the return parameter of done, info, reward, and obs
        # Calculation of reward function too i.e. total bandwidth provided to the user
        new_reward = sum(sum(user_asso_flag))
        reward = new_reward - previous_reward
        previous_reward = new_reward

        if flag != 0:
            isDone = True
            reward -= 10

        # Return of obs, reward, done, info
        return np.copy(self.state).reshape(1, self.NUM_UAV * 3), reward, isDone, "empty"

    def render(self, ax, mode='human', close=False):
        # implement viz
        if mode == 'human':
            ax.cla()
            position = self.state[:, 0:2] * self.grid_space
            ax.scatter(self.u_loc[:, 0], self.u_loc[:, 1], c = '#ff0000', marker='o', label = "Users")
            ax.scatter(position[:, 0], position[:, 1], c = '#000000', marker='x', label = "UAV")
            for (i,j) in (position[:,:]):
                cc = plt.Circle((i,j), self.coverage_radius, alpha=0.1)
                ax.set_aspect(1)
                ax.add_artist(cc)
            ax.legend()
            plt.pause(0.5)
            plt.xlim(-50, 1050)
            plt.ylim(-50, 1050)
            plt.draw()

    def reset(self):
        # reset out states
        # set the states to the hotspots and one at the centre for faster convergence
        # further complexity by choosing random value of state
        # self.state[:, 0:2] = [[1, 2], [4, 2], [7, 3], [3, 8], [4, 5]]
        self.state[:, 0:2] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.state[:, 2] = self.UAV_HEIGHT
        return self.state

    def get_state(self):
        state_loc = np.zeros((self.NUM_UAV, 2))
        for k in range(self.NUM_UAV):
            state_loc[k, 0] = self.state[k, 0]
            state_loc[k, 1] = self.state[k, 1]
        return state_loc

