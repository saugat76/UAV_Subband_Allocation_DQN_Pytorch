from scipy.io import savemat, loadmat
import os
import matplotlib.pyplot as plt

os.chdir(r'Results\Result_11_08\5_UAV\Level_1_Implicit_Info_Exchange\Run_001')
episode_reward_1 = loadmat('episodic_reward.mat')
# print(episode_reward_1)
num_episode = list(episode_reward_1['num_episode'])
episode_reward = list(episode_reward_1['episodic_reward'])
num_episode = num_episode[0]
episode_reward = episode_reward[0]
fig = plt.figure()
plt.plot(num_episode, episode_reward)
print(num_episode)
print(episode_reward)
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Episode vs Episodic Reward")
plt.show()