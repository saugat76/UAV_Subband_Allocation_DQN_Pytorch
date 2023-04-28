###################################
## Environment Setup of for UAV  ##
################################### 

import gym
import numpy as np
import math
import matplotlib.pyplot as plt
import time 

###################################
##     UAV Class Defination      ##
###################################

class UAVenv(gym.Env):
    """Custom Environment that follows gym interface """
    metadata = {'render.modes': ['human']}

    ## Polar to Cartesian and vice versa
    def pol2cart(r,theta):
        return (r * np.cos(theta), r * np.sin(theta))

    def cart2pol(z):
            return (np.abs(z), np.angle(z))

    ############################################################################
    ##     First User Distribution // Hotspots with Uniform Distribution      ##
    ############################################################################
    # User distribution on the target area // NUM_USER/5 users in each of four hotspots
    # Remaining NUM_USER/5 is then uniformly distributed in the target area
    # HOTSPOTS = np.array(
    #     [[200, 200], [800, 800], [300, 800], [800, 300]])  # Position setup in grid size rather than actual distance
    # USER_DIS = int(NUM_USER / NUM_UAV)
    # USER_LOC = np.zeros((NUM_USER - USER_DIS, 2))
    
    # for i in range(len(HOTSPOTS)):
    #     for j in range(USER_DIS):
    #         temp_loc_r = random.uniform(-(1/3.5)*COVERAGE_XY, (1/3.5)*COVERAGE_XY)
    #         temp_loc_theta = random.uniform(0, 2*math.pi)
    #         temp_loc = pol2cart(temp_loc_r, temp_loc_theta)
    #         (temp_loc_1, temp_loc_2) = temp_loc
    #         temp_loc_1 = temp_loc_1 + HOTSPOTS[i, 0]
    #         temp_loc_2 = temp_loc_2 + HOTSPOTS[i, 1]
    #         USER_LOC[i * USER_DIS + j, :] = [temp_loc_1, temp_loc_2]
    # temp_loc = np.random.uniform(low=0, high=COVERAGE_XY, size=(USER_DIS, 2))
    # USER_LOC = np.concatenate((USER_LOC, temp_loc))
    # np.savetxt('UserLocation.txt', USER_LOC, fmt='%.3e', delimiter=' ', newline='\n')

    # Saving the user location on a file instead of generating everytime

    USER_LOC = np.loadtxt('UserLocation.txt', delimiter=' ').astype(np.int64)


    #############################################################################
    ##     Second User Distribution // Hotspots with Uniform Distribution      ##
    #############################################################################
    # USER_LOC = np.random.uniform(low=0, high=COVERAGE_XY, size=(NUM_USER, 2))
    # np.savetxt('UserLocation_Uniform.txt', USER_LOC, fmt='%.3e', delimiter=' ', newline='\n')

    # Saving the user location on a file instead of generating everytime

    # USER_LOC = np.loadtxt('UserLocation_Uniform.txt', delimiter=' ').astype(np.int64)
    # User RB requirement // currently based randomly, can be later calculated using SINR value and Shannon Capacity Theorem
    # USER_RB_REQ = np.random.randint(low=1, high=3, size=(NUM_USER, 1))
    # USER_RB_REQ[np.random.randint(low = 0, high=NUM_USER, size=(NUM_USER,1))] = 1
    # print(sum(USER_RB_REQ))
    # np.savetxt('UserRBReq.txt', USER_RB_REQ, delimiter=' ', newline='\n')
    
    USER_RB_REQ = np.loadtxt('UserRBReq.txt', delimiter=' ').astype(np.int64)

    def __init__(self, args):
        super(UAVenv, self).__init__()
         
        # Environment specific params 
        self.args = args
        self.NUM_USER = self.args.num_user                      # Number of ground user
        self.NUM_UAV = self.args.num_uav                        # Number of UAV
        Fc = self.args.carrier_freq                             # Operating Frequency 2 GHz
        LightSpeed = 3 * (10 ** 8)                              # Speed of Light
        self.WaveLength = LightSpeed / (Fc * (10 ** 9))         # Wavelength of the wave
        self.COVERAGE_XY = self.args.coverage_xy
        self.UAV_HEIGHT = self.args.uav_height
        self.BS_LOC = np.zeros((self.NUM_UAV, 3))
        self.THETA = self.args.theta * math.pi / 180            # In radian  // Bandwidth for a resource block (This value is representing 2*theta instead of theta)
        self. BW_UAV = self.args.bw_uav                         # Total Bandwidth per UAV   
        self.BW_RB = self.args.bw_rb                            # Bandwidth of a Resource Block
        self.ACTUAL_BW_UAV = self.BW_UAV * 0.9
        self.grid_space = self.args.grid_space
        self.GRID_SIZE = int(self.COVERAGE_XY / self.grid_space)# Each grid defined as 100m block
        self.UAV_DIST_THRS = self.args.uav_dis_th               # Distnce that defines the term "neighbours" // UAV closer than this distance share their information
        self.dis_penalty_pri = self.args.dist_pri_param         # Priority value for defined for the distance penalty // 
                                                                # // Value ranges from 0 (overlapping UAV doesnot affect reward) to 1 (Prioritizes overlapping area as negative reward to full extent)


        # Defining action spaces // UAV RB allocation to each user increase each by 1 untill remains
        # Five different action for the movement of each UAV
        # 0 = Right, 1 = Left, 2 = straight, 3 = back, 4 = Hover
        # Defining Observation spaces // UAV RB to each user
        # Position of the UAV in space // X and Y pos                                          
        self.u_loc = self.USER_LOC
        self.state = np.zeros((self.NUM_UAV, 3), dtype=np.int32)
        # Set the states to the hotspots and one at the centre for faster convergence
        # Further complexity by choosing random value of state or starting at same initial position
        # self.state[:, 0:2] = [[1, 2], [4, 2], [7, 3], [3, 8], [4, 5]]
        # Starting UAV Position at the center of the target area
        # self.state[:, 0:2] = [[5, 5], [5, 5],[5, 5], [5, 5], [5, 5], [5, 5], [5, 5]]
        self.state[:, 0:2] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        self.coverage_radius = self.UAV_HEIGHT * np.tan(self.THETA / 2)
        self.flag = [0, 0, 0, 0, 0]
        print(self.coverage_radius)

    def step(self, action, info_exchange_lvl):
        # Take the action
        # Assignment of sub carrier band to users
        # Reshape of actions
        # Execution of one step within the environment
        # Deal with out of boundaries conditions
        isDone = False
        # Calculate the distance of every users to the UAV BS and organize as a list
        dist_u_uav = np.zeros(shape=(self.NUM_UAV, self.NUM_USER))
        for i in range(self.NUM_UAV):
            temp_x = self.state[i, 0]
            temp_y = self.state[i, 1]
            # One Step Action
            # print(action)
            if action[i] == 0:
                self.state[i, 0] = self.state[i, 0] + 1
            elif action[i] == 1:
                self.state[i, 0] = self.state[i, 0] - 1
            elif action[i] == 2:
                self.state[i, 1] = self.state[i, 1] + 1
            elif action[i] == 3:
                self.state[i, 1] = self.state[i, 1] - 1
            elif action[i] == 4:
                pass
            else:
                print("Error Action Value")

            # Take boundary condition into account // Individual flag for punishing the UAV
            if self.state[i,0] < 0 or self.state[i,0] > self.GRID_SIZE or self.state[i, 1] < 0 or self.state[i,1] > self.GRID_SIZE:
                self.state[i, 0] = temp_x
                self.state[i, 1] = temp_y
                self.flag[i] = 1                # Later penalize the reward value based on the flag
            else:
                self.flag[i] = 0

            # Calculation of the distance value for all UAV and User
            for l in range(self.NUM_USER):
                dist_u_uav[i, l] = math.sqrt((self.u_loc[l, 0] - (self.state[i, 0] * self.grid_space)) ** 2 + (self.u_loc[l, 1] -
                                                                                        (self.state[i, 1] * self.grid_space)) ** 2)
        max_rb_num = self.ACTUAL_BW_UAV / self.BW_RB

        ##############################################
        ## UAV Buffer to Penalize the Reward Value  ##
        ##############################################

        # Calculation of the distance between the UAVs
        # Based on these value another penalty of the reward will be condidered
        # This is done to increase the spacing beteween the UAVs 
        # Can be thought as the exchange of the UAVs postition information
        # As the distance between the neighbour (in this case all) UAV is use to reduce the overlapping region
        dist_uav_uav = np.zeros(shape=(self.NUM_UAV, self.NUM_UAV), dtype="float32")
        for k in range(self.NUM_UAV):
            for l in range(self.NUM_UAV):
                dist_uav_uav[k, l] = math.sqrt(((self.state[l, 0] - self.state[k, 0]) * self.grid_space) ** 2 + ((self.state[l, 1] -
                                                                                        self.state[k, 1]) * self.grid_space) ** 2)

        penalty_overlap = np.zeros(shape=(self.NUM_UAV, 1), dtype="float32")
        max_overlap_penalty = self.dis_penalty_pri *(self.NUM_USER / self.NUM_UAV)
        for k in range(self.NUM_UAV):
            temp_penalty = 0
            for l in range(self.NUM_UAV):
                if k != l:
                    temp_penalty = max(0, ((2*self.coverage_radius - dist_uav_uav[k, l]) / (2*self.coverage_radius)) * max_overlap_penalty)
                    penalty_overlap[k] += temp_penalty  

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
                close_uav = np.argmin(dist_u_uav[:,i])                    # Closest UAV index
                if dist_u_uav[close_uav, i] <= self.coverage_radius:      # UAV - User distance within the coverage radius then only connection request
                    connection_request[close_uav, i] = 1                  # All staifies, then connection request for the UAV - User   

        # Allocating only 80% of max cap in first run
        # After all the user has send their connection request,
        # UAV only admits Users closest to and if bandwidth is available
        user_asso_flag = np.zeros(shape=(self.NUM_UAV, self.NUM_USER), dtype="int")
        rb_allocated = np.zeros(shape=(self.NUM_UAV,1), dtype="int")

        for i in range(self.NUM_UAV):
            # Maximum Capacity for a single UAV
            cap_rb_num = int(0.8 * max_rb_num)
            # Sorting the users with the connection request to this UAV
            temp_user = np.where(connection_request[i, :] == 1)
            temp_user_distance = dist_u_uav[i, temp_user]
            temp_user_sorted = np.argsort(temp_user_distance) # Contains user index with closest 2D distance value (out of requested user)
            # The user list are already sorted, to associate flag bit of user upto the index from
            # The 0 is kept because numpy for some reason stores as a two dimenstional data
            # Convert temp_user to np_array so could be indexed easily
            temp_user = np.array(temp_user)
            # Actual index of the users that send connection request sorted based on the distance value
            temp_user_actual_idx = temp_user[0, temp_user_sorted]
            # Set user association flag to 1 for that UAV and closest user index
            # Iterate over the sorted user index and allocate RB is only available
            for user_index in temp_user_actual_idx[0]:
                if self.USER_RB_REQ[user_index] + rb_allocated[i] <= cap_rb_num:
                    user_asso_flag[i, user_index] = 1
                    rb_allocated[i] += self.USER_RB_REQ[user_index]
                else:
                    break

        # For the second sweep, sweep through all users
        # If the user is not associated choose the closest UAV and check whether it has any available resource
        # If so allocate the resource and set the User association flag bit of that user to 1
        for j in range(self.NUM_USER):
            if not(np.any(user_asso_flag[:, j] != 0)):
                close_uav_id = dist_u_uav[:, j]
                close_uav_id = [i[0] for i in sorted(enumerate(close_uav_id), key=lambda x: x[1])]
                for close_id in close_uav_id:
                    if dist_u_uav[close_id, j] <= self.coverage_radius:
                        if np.sum(rb_allocated[close_id]) < max_rb_num:
                            rb_allocated[close_id] += self.USER_RB_REQ[j]
                            user_asso_flag[close_id, j] = 1
                            break
        
        # Need to work on the return parameter of done, info, reward, and obs
        # Calculation of reward function too i.e. total bandwidth providednew to the user
        # Using some form of weighted average to do the reward calculation instead of the collective reward value only
        ################################################################
        ##     Opt.1  No. of User Connected as Indiviudal Reward      ##
        ################################################################
        if info_exchange_lvl == 1 or info_exchange_lvl == 4:
            sum_user_assoc = np.sum(user_asso_flag, axis = 1)
            reward_solo = np.zeros(np.size(sum_user_assoc), dtype="float32")
            for k in range(self.NUM_UAV):
                if self.flag[k] != 0:
                    reward_solo[k] = np.copy(sum_user_assoc[k] - 2)
                    isDone = True
                else:
                    reward_solo[k] = np.copy(sum_user_assoc[k]) 
            reward = np.copy(reward_solo)

        #############################################################################################
        ##     Opt.2  No. of User Connected as Indiviudal Reward with Penalty Over Buffer Area     ##
        #############################################################################################
        elif info_exchange_lvl == 3:
            sum_user_assoc = np.sum(user_asso_flag, axis = 1)
            reward_solo = np.zeros(np.size(sum_user_assoc), dtype = "float32")
            penalty_overlap = penalty_overlap.flatten()
            for k in range(self.NUM_UAV):
                if self.flag[k] != 0:
                    reward_solo[k] = np.copy(sum_user_assoc[k] - 2) - penalty_overlap[k]
                    isDone = True
                else:
                    reward_solo[k] = (sum_user_assoc[k] - penalty_overlap[k])
            # Calculation of reward based in the change in the number of connected user
            reward = np.copy(reward_solo)

        # Collective reward exchange of nuumber of user associated and calculation of the reward based on it
        # Only share the information to the neighbours based on distance values
        ################################################################
        ##     Opt.3  No. of User Connected as Collective Reward      ##
        ################################################################
        elif info_exchange_lvl == 2:    
            sum_user_assoc = np.sum(user_asso_flag, axis = 1)
            sum_user_assoc_temp = np.copy(sum_user_assoc)
            reward_ind = np.zeros(np.size(sum_user_assoc))
            reward = 0
            for k in range(self.NUM_UAV):
                if self.flag[k] != 0:
                    temp_user_id = np.where(dist_uav_uav[k, :] <= self.UAV_DIST_THRS)
                    reward_ind[k] = np.average(sum_user_assoc_temp[temp_user_id])
                    reward_ind[k] -= 2
                    isDone = True
                else:
                    temp_user_id = np.where(dist_uav_uav[k, :] <= self.UAV_DIST_THRS)
                    reward_ind[k] = np.average(sum_user_assoc[temp_user_id])
            reward = np.copy(reward_ind)

        
        # Defining the reward function by the number of covered user
        ################################################################
        ##            Opt.4  No. of User Covered as Reward            ##
        ################################################################
        # reward = np.copy(total_user_covered)

        # Return of obs, reward, done, info
        return np.copy(self.state).reshape(1, self.NUM_UAV * 3), reward, isDone, "empty", sum_user_assoc, rb_allocated


    def render(self, ax, mode='human', close=False):
        # Implement viz
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
        # Reset out states
        # Set the states to the hotspots and one at the centre for faster convergence
        # Further complexity by choosing random value of state
        # self.state[:, 0:2] = [[1, 2], [4, 2], [7, 3], [3, 8], [4, 5]]
        self.state[:, 0:2] = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
        # Starting UAV Position at the center of the target area
        # self.state[:, 0:2] = [[5, 5], [5, 5],[5, 5], [5, 5], [5, 5], [5, 5],[5, 5]]
        self.state[:, 2] = self.UAV_HEIGHT
        return self.state

    def get_state(self):
        state_loc = np.zeros((self.NUM_UAV, 2))
        for k in range(self.NUM_UAV):
            state_loc[k, 0] = self.state[k, 0]
            state_loc[k, 1] = self.state[k, 1]
        return state_loc