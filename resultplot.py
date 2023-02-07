from scipy.io import savemat, loadmat
import os
import matplotlib.pyplot as plt
import matplotlib

font = {'family': 'Times New Roman',
        'size' : 12}

matplotlib.rc('font', **font)


###############################################################
##  Differernt level of information sharing for NUM_UAV = 5  ##
###############################################################

## Use of Run_002 Hyperparamters
# Batch Size = 512
# DNN Epoch = 1
# Update Rate = 10
# Number Episode = 351
# Number Epoch = 100
# Epsilon = 0.1 // Constant Value Used 
# Aplha / Learning Rate = 3.5e-4

## Plot for data from NUM_UAV - 5
episode_reward_level_1 = loadmat(r'Results\Result_11_08\5_UAV\Level_1_Implicit_Info_Exchange\Run_002\episodic_reward.mat')
episode_reward_level_2 = loadmat(r'Results\Result_11_08\5_UAV\Level_2_Reward_Info_Exchange\Run_002\episodic_reward.mat')
episode_reward_level_3 = loadmat(r'Results\Result_11_08\5_UAV\Level_3_Position_of_UAV_(Distance_Penalty)\Run_002\episodic_reward.mat')
episode_reward_level_4 = loadmat(r'Results\Result_11_08\5_UAV\Level_4_Drone_State_Space_Exchange\Run_002\episodic_reward.mat')


num_episode_1 = list(episode_reward_level_1['num_episode'])
episode_reward_1 = list(episode_reward_level_1['episodic_reward'])
num_episode_2 = list(episode_reward_level_2['num_episode'])
episode_reward_2 = list(episode_reward_level_2['episodic_reward'])
num_episode_3 = list(episode_reward_level_3['num_episode'])
episode_reward_3 = list(episode_reward_level_3['episodic_reward'])
num_episode_4 = list(episode_reward_level_4['num_episode'])
episode_reward_4 = list(episode_reward_level_4['episodic_reward'])


num_episode_1 = num_episode_1[0]
episode_reward_1 = episode_reward_1[0]
num_episode_2 = num_episode_2[0]
episode_reward_2 = episode_reward_2[0]
num_episode_3 = num_episode_3[0]
episode_reward_3 = episode_reward_3[0]
num_episode_4 = num_episode_4[0]
episode_reward_4 = episode_reward_4[0]


fig = plt.figure()
plt.plot(num_episode_1[0:350], episode_reward_1[0:350], 'r', label='Level 1: Implicit')
plt.plot(num_episode_2[0:350], episode_reward_2[0:350], 'g', label='Level 2: Reward exchange')
plt.plot(num_episode_3, episode_reward_3, 'b', label='Level 3: Distance penalty')
plt.plot(num_episode_4, episode_reward_4, 'm', label='Level 4: State')
plt.legend(loc="lower right")
plt.xlabel("Episode", fontsize = 14, family = 'Times New Roman')
plt.ylabel("Episodic reward", fontsize = 14, family = 'Times New Roman')
plt.title("Episode vs episodic reward with run002 parameters")
plt.show()

# ## Plot for data from NUM_UAV - 7 and NUM_UAV - 5 Compari
# episode_reward_7_level_4 = loadmat(r'Results\Result_11_08\7_UAV\Level_4_Drone_State_Space_Exchange\Run_002\episodic_reward.mat')
# episode_reward_5_level_4 = loadmat(r'Results\Result_11_08\5_UAV\Level_4_Drone_State_Space_Exchange\Run_002\episodic_reward.mat')


# num_episode_7_4 = list(episode_reward_7_level_4['num_episode'])
# episode_reward_7_4 = list(episode_reward_7_level_4['episodic_reward'])
# num_episode_5_4 = list(episode_reward_5_level_4['num_episode'])
# episode_reward_5_4 = list(episode_reward_5_level_4['episodic_reward'])

# num_episode_7_4 = num_episode_7_4[0]
# episode_reward_7_4 = episode_reward_7_4[0]
# num_episode_5_4 = num_episode_5_4[0]
# episode_reward_5_4 = episode_reward_5_4[0]


# fig = plt.figure()
# plt.plot(num_episode_5_4, episode_reward_5_4, 'r', label='Num UAV = 5')
# plt.plot(num_episode_7_4, episode_reward_7_4, 'g', label='Num UAV = 7')

# plt.legend(loc="lower right")
# plt.xlabel("Episode")
# plt.ylabel("Episodic Reward")
# plt.title("Episode vs Episodic Reward with 002 Hyperparameters")
# plt.show()


# #################################################################
# #### Level 3  Differernt Distance Penalty Priority  UAV 7   ####
# #################################################################
# episode_reward_16 = loadmat(r'Results\Results_11_15\7_UAV\Distance Threshold Neighbours\Distance_Threshold1000\Level_3_Position_of_UAV_(Distance_Penalty)\Starting_Pos(0,0)\Run_004\episodic_reward.mat')
# episode_reward_14 = loadmat(r'Results\Results_11_15\7_UAV\Distance Threshold Neighbours\Distance_Threshold1000\Level_3_Position_of_UAV_(Distance_Penalty)\Starting_Pos(0,0)\Run_005\episodic_reward.mat')

# num_episode_16 = list(episode_reward_16['num_episode'])
# episode_reward_16 = list(episode_reward_16['episodic_reward'])
# num_episode_14 = list(episode_reward_14['num_episode'])
# episode_reward_14 = list(episode_reward_14['episodic_reward'])


# num_episode_16 = num_episode_16[0]
# episode_reward_16 = episode_reward_16[0]
# num_episode_14 = num_episode_14[0]
# episode_reward_14 = episode_reward_14[0]


# # fig = plt.figure()
# plt.plot(num_episode_16, episode_reward_16, 'r', label='Distance_Penalty_Priority = 1/6')
# plt.plot(num_episode_14, episode_reward_14, 'g', label='Distance Penalty Priority = 1/4')

# plt.legend(loc="lower right")
# plt.xlabel("Episode")
# plt.ylabel("Episodic Reward")
# plt.title("Episode vs Episodic Reward for UAV 7")
# plt.show()


# ###############################################################
# #### Level 3 Differernt NN Structure with all same   ####
# ###############################################################
# episode_reward_16 = loadmat(r'Results\Results_11_15\7_UAV\Distance Threshold Neighbours\Distance_Threshold1000\Level_3_Position_of_UAV_(Distance_Penalty)\Starting_Pos(0,0)\Run_005\episodic_reward.mat')
# episode_reward_14 = loadmat(r'Results\Results_11_15\7_UAV\Distance Threshold Neighbours\Distance_Threshold1000\Level_3_Position_of_UAV_(Distance_Penalty)\Starting_Pos(0,0)\Run_006\episodic_reward.mat')

# num_episode_16 = list(episode_reward_16['num_episode'])
# episode_reward_16 = list(episode_reward_16['episodic_reward'])
# num_episode_14 = list(episode_reward_14['num_episode'])
# episode_reward_14 = list(episode_reward_14['episodic_reward'])


# num_episode_16 = num_episode_16[0]
# episode_reward_16 = episode_reward_16[0]
# num_episode_14 = num_episode_14[0]
# episode_reward_14 = episode_reward_14[0]


# # fig = plt.figure()
# plt.plot(num_episode_16, episode_reward_16, 'r', label='NN 2 Layered 400 Nodes')
# plt.plot(num_episode_14, episode_reward_14, 'b', label='NN 3 Layered 300 Nodes')

# plt.legend(loc="lower right")
# plt.xlabel("Episode")
# plt.ylabel("Episodic Reward")
# plt.title("Episode vs Episodic Reward for UAV 7")
# plt.show()

# ###############################################################
# ####  Level 4 State Space Sharing Different Start Positn   ####
# ###############################################################
# episode_reward_16 = loadmat(r'C:\Users\tripats\Documents\Results\Results_11_15\7_UAV\Distance Threshold Neighbours\Distance_Threshold1000\Level_4_Drone_State_Space_Exchange\Starting_Pos(0,0)\episodic_reward.mat')
# episode_reward_14 = loadmat(r'C:\Users\tripats\Documents\Results\Results_11_15\7_UAV\Distance Threshold Neighbours\Distance_Threshold1000\Level_4_Drone_State_Space_Exchange\Starting_Pos(5,5)\episodic_reward.mat')

# num_episode_16 = list(episode_reward_16['num_episode'])
# episode_reward_16 = list(episode_reward_16['episodic_reward'])
# num_episode_14 = list(episode_reward_14['num_episode'])
# episode_reward_14 = list(episode_reward_14['episodic_reward'])


# num_episode_16 = num_episode_16[0]
# episode_reward_16 = episode_reward_16[0]
# num_episode_14 = num_episode_14[0]
# episode_reward_14 = episode_reward_14[0]


# # fig = plt.figure()
# plt.plot(num_episode_16, episode_reward_16, 'tab:orange', label='Starting Position (0, 0)')
# plt.plot(num_episode_14, episode_reward_14, 'tab:cyan', label='Starting Position (5, 5)')

# plt.legend(loc="lower right")
# plt.xlabel("Episode")
# plt.ylabel("Episodic Reward")
# plt.title("Episode vs Episodic Reward for UAV 7 : Level 4")
# plt.show()

# ###############################################################
# ####  Level 3 Different Start Positn   ####
# ###############################################################
# episode_reward_16 = loadmat(r'C:\Users\tripats\Documents\Results\Results_11_15\7_UAV\Distance Threshold Neighbours\Distance_Threshold1000\Level_3_Position_of_UAV_(Distance_Penalty)\Starting_Pos(0,0)\Run_006\episodic_reward.mat')
# episode_reward_14 = loadmat(r'C:\Users\tripats\Documents\Results\Results_11_15\7_UAV\Distance Threshold Neighbours\Distance_Threshold1000\Level_3_Position_of_UAV_(Distance_Penalty)\Statrting_Pos(5,5)\Run_006\episodic_reward.mat')

# num_episode_16 = list(episode_reward_16['num_episode'])
# episode_reward_16 = list(episode_reward_16['episodic_reward'])
# num_episode_14 = list(episode_reward_14['num_episode'])
# episode_reward_14 = list(episode_reward_14['episodic_reward'])


# num_episode_16 = num_episode_16[0]
# episode_reward_16 = episode_reward_16[0]
# num_episode_14 = num_episode_14[0]
# episode_reward_14 = episode_reward_14[0]


# # fig = plt.figure()
# plt.plot(num_episode_16, episode_reward_16, 'maroon', label='Starting Position (0, 0)')
# plt.plot(num_episode_14, episode_reward_14, 'darkviolet', label='Starting Position (5, 5)')

# plt.legend(loc="lower right")
# plt.xlabel("Episode")
# plt.ylabel("Episodic Reward")
# plt.title("Episode vs Episodic Reward for UAV 7 : Level 3")
# plt.show()