from ctypes.wintypes import tagRECT
from os import stat_result, times_result
import random
from re import S
import re
from sre_parse import State
from ssl import ALERT_DESCRIPTION_USER_CANCELLED
from tabnanny import verbose
import numpy as np
from collections import defaultdict
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from torch import ne, tensor
from uav_env import UAVenv
from misc import final_render
import gym 
from collections import deque
import torch
from torch import Tensor, nn 
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

## GPU configuration use for faster processing
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cpu"

# DNN modeling
class NeuralNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(NeuralNetwork, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.linear_stack = model = nn.Sequential(
            nn.Linear(self.state_size,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64, self.action_size)
        ).to(device=device)

    def forward(self, x):
        x = x.to(device)
        Q_values = self.linear_stack(x)
        return Q_values


class DQL:
    # Initializing a Deep Neural Network
    def __init__(self):
        self.state_size = 2
        self.action_size = 5
        self.replay_buffer = deque(maxlen = 125000)
        self.gamma = 0.90
        self.epsilon = 0.1
        self.learning_rate = 0.0075
        self.main_network = NeuralNetwork(self.state_size, self.action_size).to(device)
        self.target_network = NeuralNetwork(self.state_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr = self.learning_rate)
        self.loss_func = nn.MSELoss()

    # Storing information of individual UAV information in their respective buffer
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    # Deployment of epsilon greedy policy
    def epsilon_greedy(self, state):
        temp = random.random()
        if temp <= self.epsilon:
            action = np.random.randint(0, 4)
        else:
            state = torch.unsqueeze(torch.FloatTensor(state),0)
            # print(state)
            Q_values = self.main_network.forward(state)
            # print(Q_values)
            action = torch.max(Q_values.cpu(), 1)[1].data.numpy()
            # print(action)
            action = action[0]
        return action

    # Training of the DNN 
    def train(self,batch_size, dnn_epoch, batch_size_internal):
        # internal_buffer = random.sample(self.replay_buffer, batch_size)
        for k in range(dnn_epoch):
            minibatch = random.sample(self.replay_buffer, batch_size_internal)
            minibatch = np.vstack(minibatch)
            minibatch = minibatch.reshape(batch_size_internal,5)
            state = torch.FloatTensor(np.vstack(minibatch[:,0]))
            # print(state)
            action = torch.LongTensor(np.vstack(minibatch[:,1]))
            reward = torch.FloatTensor(np.vstack(minibatch[:,2]))
            next_state = torch.FloatTensor(np.vstack(minibatch[:,3]))
            done = torch.Tensor(np.vstack(minibatch[:,4]))
            state = state.to(device = device)
            action = action.to(device = device)
            reward = reward.to(device = device)
            next_state = next_state.to(device = device)
            done = done.to(device = device)

            Q_next = self.target_network(next_state).detach()
            target_Q = reward.cpu().squeeze() + self.gamma * Q_next.cpu().max(1)[0].view(batch_size_internal, 1).squeeze() * (
                1 - np.array([state[e].cpu().mean() == next_state[e].cpu().mean() for e in range(len(next_state))])
            ) 
            target_Q = target_Q.float()
            Q_main = self.main_network(state).gather(1, action).squeeze()

            loss = self.loss_func(Q_main.cpu(), target_Q.cpu().detach())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 200
num_epochs = 100
discount_factor = 0.90
alpha = 0.5
batch_size = 512
batch_size_internal = 512
update_rate = 10  #50
dnn_epoch = 1
train_freq = 32 #NA
random.seed(10)

# Keeping track of the episode reward
episode_reward = np.zeros(num_episode)

fig = plt.figure()
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])

UAV_OB = [None, None, None, None, None]


for k in range(NUM_UAV):
            UAV_OB[k] = DQL()

best_result = 0

for i_episode in range(num_episode):
    print(i_episode)

    # Environment reset and get the states
    u_env.reset()

    # Get the initial states
    states = u_env.get_state()

    
    for t in range(num_epochs):
        drone_act_list = []

        # Update the target network 
        for k in range(NUM_UAV):
            if t % update_rate == 0:
                UAV_OB[k].target_network.load_state_dict(UAV_OB[k].main_network.state_dict())
                
        # Determining the actions for all drones
        # print(states)
        states_ten = torch.from_numpy(states)
        # print(states_ten)
        for k in range(NUM_UAV):
            state = states_ten[k, :]
            # print(state)                
            action = UAV_OB[k].epsilon_greedy(state.float())
            # print(action)
            drone_act_list.append(action + 1)
            # print('new')

        # print(drone_act_list)

        # Find the global reward for the combined set of actions for the UAV
        # print(drone_act_list)
        temp_data = u_env.step(drone_act_list)
        reward = temp_data[1]
        done = temp_data[2]
        next_state = u_env.get_state()

        # Store the transition information
        for k in range(NUM_UAV):
                state = states_ten[k, :]
                action = drone_act_list[k] - 1
                next_sta = next_state[k, :]
                UAV_OB[k].store_transition(state, action, reward, next_sta, done)

        episode_reward[i_episode] += reward

        states = next_state

        # if done:
        #     break

        for k in range(NUM_UAV):
            if len(UAV_OB[k].replay_buffer) > batch_size:
                UAV_OB[k].train(batch_size, dnn_epoch, batch_size_internal)

    if i_episode % 10 == 0:
        # Reset of the environment
        u_env.reset()
        # Get the states
        # Get the states
        states = u_env.get_state()
        states_ten = torch.from_numpy(states)
        for t in range(100):
            drone_act_list = []
            for k in range(NUM_UAV):
                state = states[k,:]
                state = torch.unsqueeze(torch.FloatTensor(state),0)
                Q_values = UAV_OB[k].main_network.forward(state.float())
                # print(Q_values)
                best_next_action = torch.max(Q_values.cpu(), 1)[1].data.numpy()
                best_next_action = best_next_action[0]
                drone_act_list.append(best_next_action + 1)
            temp_data = u_env.step(drone_act_list)
            states = u_env.get_state()
            states_fin = states
        u_env.render(ax1)
        plt.title("Intermediate state of UAV in this episode")
        print(drone_act_list)
        print("Number of user connected in ",i_episode," episode is: ", temp_data[4])
        if best_result < temp_data[4]:
            best_result = temp_data[4]
            best_state = states



def smooth(y, pts):
    box = np.ones(pts)/pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


# Plot the accumulated reward vs episodes
fig = plt.figure()
plt.plot(range(0, num_episode), episode_reward)
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Episode vs Episodic Reward")
plt.show()
fig = plt.figure()
smoothed = smooth(episode_reward, 10)
plt.plot(range(0, num_episode-10), smoothed[0:len(smoothed)-10] )
plt.show()
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Smoothed Epidode vs Episodic Reward")
fig = plt.figure()
final_render(states_fin, "final")
fig = plt.figure()
final_render(best_state, "best")
# mdict = {'Q': Q_values}
# savemat('Q.mat', mdict)
print(states_fin)
print('Total Connected User in Final Stage', temp_data[4])
print("Best State")
print(best_state)
print("Total Connected User (Best Outcome)", best_result)