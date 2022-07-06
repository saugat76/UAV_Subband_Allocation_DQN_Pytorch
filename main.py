from ctypes.wintypes import tagRECT
from os import stat_result, times_result
import random
from re import S
from sre_parse import State
from ssl import ALERT_DESCRIPTION_USER_CANCELLED
from tabnanny import verbose
import numpy as np
from collections import defaultdict
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from pygame import bufferproxy
from scipy.io import savemat
from scipy.signal import savgol_filter
from torch import ne
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DNN modeling
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.state_size = 2
        self.action_size = 5
        self.linear_stack = model = nn.Sequential(
            nn.Linear(self.state_size,20),
            nn.Linear(20,20),
            nn.Linear(20, self.action_size)
        )

    def forward(self, x):
        logits = self.linear_stack(x)
        return logits


class DQL:
    # Initializing a Deep Neural Network
    def __init__(self):
        self.state_size = 2
        self.action_size = 5
        self.replay_buffer = deque(maxlen = 10000)
        self.gamma = 0.90
        self.epsilon = 0.1
        self.update_rate = 20
        self.main_network = NeuralNetwork()
        self.main_network.to(device)
        self.target_network = NeuralNetwork()
        self.target_network.to(device)
        # self.target_network.set_weights(self.main_network.get_weights())

    # DNN modeling
    class NeuralNetwork(nn.Module):
        def __init__(self):
            super(NeuralNetwork, self).__init__()
            self.linear_stack = model = nn.Sequential(
                nn.Linear(self.state_size,20),
                nn.Linear(20,20),
                nn.Linear(20, self.action_size)
            )
    
        def forward(self, x):
            logits = self.linear_stack(x)
            return logits
    
    # Storing information of individual UAV information in their respective buffer
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    # Deployment of epsilon greedy policy
    def epsilon_greedy(self, state):   
        temp = random.random()
        if temp <= self.epsilon:
            action = random.randint(0, 4)
        else:
            Q_values = self.main_network.predict(state, verbose = 0)
            # print(Q_values)
            action = np.argmax(Q_values[0])
        return action

    # Training of the DNN 
    def train(self,batch_size_internal):
        Q_val_full = np.array([])
        states_full = np.array([])
        minibatch = random.sample(self.replay_buffer, batch_size_internal)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                next_state = next_state.reshape(1,2)
                target_Q = (reward + self.gamma * np.argmax(self.target_network.predict(next_state, verbose=0)))
            else:
                target_Q = reward
            state = state.reshape(1,2)
            Q_values = self.main_network.predict(state, verbose=0)
            Q_values[0][action] = target_Q
            Q_val_full = np.append(Q_val_full, Q_values[0].tolist())
            states_full = np.append(states_full, state.tolist())
        Q_val_full = np.reshape(Q_val_full, (32,5))
        states_full = np.reshape(states_full,(32,2))
        # Wrap data in Dataset objects.
        train_data = tf.data.Dataset.from_tensor_slices((states_full, Q_val_full))
        # The batch size must now be set on the Dataset objects.
        batch = 32
        train_data = train_data.batch(batch)
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        train_data = train_data.with_options(options)
        self.main_network.fit(train_data, epochs=1, verbose=1, use_multiprocessing=True, callbacks=self.callbacks)


    # Updating the weights of the target network from the main network
    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())


u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 20
num_epochs = 20
discount_factor = 0.90
alpha = 0.5
epsilon = 0.1
batch_size = 32
update_rate = 20

random.seed(10)

# Keeping track of the episode reward
episode_reward = np.zeros(num_episode)

fig = plt.figure()
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])

UAV_OB = [None, None, None, None, None]


for k in range(NUM_UAV):
            UAV_OB[k] = DQL()


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
                UAV_OB[k].update_target_network()
                
        # Determining the actions for all drones
        for k in range(NUM_UAV):
                state = states[k, :]
                # print(state)
                state = state.reshape(1,2)
                action = UAV_OB[k].epsilon_greedy(state)
                drone_act_list.append(action + 1)
    
        # Find the global reward for the combined set of actions for the UAV
        # print(drone_act_list)
        temp_data = u_env.step(drone_act_list)
        reward = temp_data[1]
        done = temp_data[2]
        next_state = u_env.get_state()

        # Store the transition information
        for k in range(NUM_UAV):
                state = states[k, :]
                action = drone_act_list[k] - 1
                next_sta = next_state[k, :]
                UAV_OB[k].store_transition(state, action, reward, next_sta, done)

        episode_reward[i_episode] += reward

        states = next_state

        for k in range(NUM_UAV):
            if len(UAV_OB[k].replay_buffer) > batch_size:
                # with strategy.scope():
                UAV_OB[k].train(batch_size)

    if i_episode % 10 == 0:
        # Reset of the environment
        u_env.reset()
        # Get the states
        # Get the states
        states = u_env.get_state()
        for t in range(100):
            drone_act_list = []
            for k in range(NUM_UAV):
                state = states[k, :]
                state = state.reshape(1,2)
                Q_values = UAV_OB[k].main_network.predict(state, verbose=0)
                # print(Q_values)
                best_next_action = np.argmax(Q_values[0])
                drone_act_list.append(best_next_action + 1)
            temp_data = u_env.step(drone_act_list)
            states = u_env.get_state()
            states_fin = states
        u_env.render(ax1)


def smooth(y, pts):
    box = np.ones(pts)/pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

# Plot the accumulated reward vs episodes
fig = plt.figure()
plt.plot(range(0, num_episode), episode_reward)
plt.show()
fig = plt.figure()
smoothed = smooth(episode_reward, 10)
plt.plot(range(0, num_episode-10), smoothed[0:len(smoothed)-10] )
plt.show()
final_render(states_fin)

# mdict = {'Q': Q_values}
# savemat('Q.mat', mdict)
print(states_fin)
print('Total Connected User in Final Stage', reward)