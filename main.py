from ctypes.wintypes import tagRECT
from os import times_result
import random
from re import S
from sre_parse import State
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
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense, Activation, Flatten
from tensorflow.keras.optimizers import Adam

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


class DQL:

    def __init__(self):
        self.state_size = 2
        self.action_size = 5
        self.replay_buffer = deque(maxlen = 5000)
        self.gamma = 0.85
        self.epsilon = 0.1
        self.update_rate = 0.1
        self.main_network = self.build_network()
        self.target_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def build_network(self):
        model = Sequential()
        model.add(Dense(20, input_shape=(self.state_size,)))
        model.add(Dense(20))
        model.add(Dense(self.action_size))
        model.compile(loss = 'MSE', optimizer = 'Adam')
        return model 
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    def epsilon_greedy(self, state):
        temp = random.random()
        if temp <= self.epsilon:
            action = random.randint(0, 4)
            action = action + 1
        else:
            Q_values = self.main_network.predict(state)
            action = np.argmax(Q_values[0]) + 1
        return action

    def train(self,batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.argmax(self.target_network.predict(next_state)))
            else:
                target_Q = reward
            Q_values = self.main_network.predict(state)
            Q_values[0][action] = target_Q
            self.main_network.fit(state, Q_values, epoch=1, verbose=0)

    def update_target_network(self):
        self.target_network.set_weights(self.main_network.get_weights())


u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 500
num_epochs = 500
discount_factor = 0.85
alpha = 0.5
epsilon = 0.1
batch_size = 2000

random.seed(10)

# Keeping track of the episode reward
episode_reward = np.zeros(num_episode)

fig = plt.figure()
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[0:1, 0:1])

UAV_OB = [None, None, None, None, None]

for k in range(NUM_UAV):
            UAV_OB[k] = DQL()

print(UAV_OB)

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
            if t % UAV_OB[k].update_rate == 0:
                UAV_OB[k].update_target_network()

        # Determining the actions for all drones
        for k in range(NUM_UAV):
            state = states[k,k]
            print(UAV_OB[k])
            action = UAV_OB[k].epsilon_greedy([[1], [5]])
            drone_act_list.append(action)
    
        # Find the global reward for the combined set of actions for the UAV
        temp_data = u_env.step(drone_act_list)
        reward = temp_data[1]
        done = temp_data[2]
        next_state = u_env.get_state()

        # Store the transition information
        for k in range(NUM_UAV):
            UAV_OB[k].store_transition(state, action, reward, next_state, done)

        episode_reward[i_episode] += reward

        states = next_state

        for k in range(NUM_UAV):
            if len(DQL.replay_buffer) > batch_size:
                DQL.train(batch_size)


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
final_render(state)
