
from ctypes.wintypes import tagRECT
import random
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

u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
state_size = (GRID_SIZE, GRID_SIZE, NUM_UAV)
action_size = u_env.action_space.n
Q_values = None


class DQN:
    def __init__(self, state_size, action_size):
        self.action_size_single =  1
        self.state_size = state_size 
        self.replay_buffer = np.zeros(shape = 5, dtype=deque(maxlen = 5000))
        self.gamma = 0.9
        self.epsilon = 0.8
        self.update_rate = 1000
        self.main_network = self.build_network()
        self.target_network.set_weights(self.main_network.get_weights())

    def build_network(self):
        model = Sequential
        model.add(Dense(100, input_shape=(self.state_size)))
        model.add(Dense(80))
        model.add(Dense(80))
        model.add(Dense(self.action_size))
        model.compile(loss = 'MSE', optimizer = 'Adam')
        return model 
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def epsilon_greedy(self, state):
        drone_act_list = []
        # Determining the actions for all drones
        for k in range(NUM_UAV):
            temp = random.random()
            if temp <= self.epsilon:
                action = random.randint(0, 4)
                action = action + 1
            else:
                Q_values[k] = self.main_network.predict(state)
                action = np.argmax(Q[0][k])
            drone_act_list.append(action)

    def train(self, batch_size):
        minibatch = random.sample(self.replay_buffer, batch_size)
        for state, action, reward, next_state, done in minibatch:
            if not done:
                target_Q = (reward + self.gamma * np.argmax(self.target_network.predict(next_state)))
            else:
                target_Q = reward
            Q_values = self.main_network.predict(state)
            Q_values[0] = target_Q
            self.main_network.fit(state, Q_values, epoch=1, verbose=0)

    def update_target_network(self):
        self.update_target_network.set_weight(self.main_network.get_weights())


num_episode = 500
num_epochs = 500
discount_factor = 0.85
alpha = 0.5
epsilon = 0.1
batch_size = 8
nun_screen = 4
dqn = DQN(state_size, action_size)
done = False

for i in range(num_episode):



        

    #     # Q = np.random.rand(NUM_UAV, 5)
    #     # main_network = None
    #     # target_network = None
    #     # for k in range(NUM_UAV):
    #     #     main_network[k] = build_network()
    #     #     target_network[k].set_weights(main_network.get_weights())


    #     # Keeping track of the episode reward
    #     episode_reward = np.zeros(num_episode)

    #     fig = plt.figure()
    #     gs = GridSpec(1, 1, figure=fig)
    #     ax1 = fig.add_subplot(gs[0:1, 0:1])


    #     for i_episode in range(num_episode):
    #         print(i_episode)
    #         # Environment reset and get the states
    #         u_env.reset()
    #         # Get the initial states
    #         states = u_env.get_state()
    #         for t in range(num_epoch):
    #             drone_act_list = []
    #             # Determining the actions for all drones
    #             for k in range(NUM_UAV):
    #                 temp = random.random()
    #                 if temp <= epsilon or i_episode == 0:
    #                     action = random.randint(0, 4)
    #                     action = action + 1
    #                 else:
    #                     Q[k] = main_network[k].predict(state)
    #                     action = np.argmax(Q[k]) + 1
    #                 drone_act_list.append(action)

    #             # Find the global reward for the combined set of actions for the UAV
    #             temp_data = u_env.step(drone_act_list)
    #             reward = temp_data[1]
    #             done = temp_data[2]
    #             next_state = u_env.get_state()
                
    #             # If done break from the loop (go to next episode)
    #             # if done:
    #             #     break

    #             # Update of the episodic reward
    #             episode_reward[i_episode] += reward
    #             # print(reward)

    #             # Use DQN update
    #             if t % update_rate == 0:
    #                 for k in range(NUM_UAV):
    #                     update_target_network(k)
                
    #             store_transition(state, action, reward, next_state, done)

    #             states = next_state

    #             if len(replay_buffer) > batch_size:
    #                 train(batch_size, uav_id)

    #         # if i_episode % 10 == 0:
    #         #     # Reset of the environment
    #         #     u_env.reset()
    #         #     # Get the states
    #         #     # Get the states
    #         #     states = u_env.get_state()
    #         #     for t in range(100):
    #         #         drone_act_list = []
    #         #         for k in range(NUM_UAV):
    #         #             best_next_action = np.argmax(Q[k][int(states[k, 0] * (GRID_SIZE + 1) + states[k, 1])]) + 1
    #         #             drone_act_list.append(best_next_action)
    #         #         temp_data = u_env.step(drone_act_list)
    #         #         states = u_env.get_state()
    #         #     u_env.render(ax1)
                

    #     return Q, episode_reward, states, reward

     # # Defining System Parameters
    # u_env = UAVenv()
    # GRID_SIZE = u_env.GRID_SIZE
    # NUM_UAV = u_env.NUM_UAV
    # NUM_USER = u_env.NUM_USER
    # num_episode = 500
    # num_epochs = 500
    # discount_factor = 0.85
    # alpha = 0.5
    # epsilon = 0.1
    # update_rate = 0.1
    # batch_size = 25


    # random.seed(10)

    # Q, episode_rewards, state, reward = dql(u_env, num_episode, num_epochs, discount_factor, alpha, epsilon, update_rate)

    # mdict = {'Q': Q}
    # savemat('Q.mat', mdict)
    # print(state)
    # print('Total Connected User in Final Stage', reward)