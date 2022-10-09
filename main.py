import random
import numpy as np
import math
from collections import defaultdict
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
from torch import ne, tensor
from uav_env import UAVenv
from misc import final_render
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
            nn.Linear(self.state_size,400),
            nn.ReLU(),
            nn.Linear(400,400),
            nn.ReLU(),
            nn.Linear(400, self.action_size)
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
        self.gamma = discount_factor
        self.epsilon = epsilon        
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = alpha
        self.main_network = NeuralNetwork(self.state_size, self.action_size).to(device)
        self.target_network = NeuralNetwork(self.state_size, self.action_size).to(device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        self.optimizer = torch.optim.Adam(self.main_network.parameters(), lr = self.learning_rate)
        self.loss_func = nn.SmoothL1Loss()      # Huber Loss // Combines MSE and MAE
        self.steps_done = 0

    # Storing information of individual UAV information in their respective buffer
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))
    
    
    # Deployment of epsilon greedy policy
    def epsilon_greedy(self, state):



        temp = random.random()
        # Epsilon decay policy is employed for faster convergence
        epsilon_thres = self.epsilon_min + (self.epsilon - self.epsilon_min) * math.exp(-1*self.steps_done/self.epsilon_decay)
        self.steps_done += 1 
        if temp <= epsilon_thres:
            action = torch.tensor([[np.random.randint(0, 4)]], device = device, dtype = torch.long)
        else:
            state = torch.unsqueeze(torch.FloatTensor(state),0)
            Q_values = self.main_network(state)
            action = Q_values.max(1)[1].view(1,1)
        return action

    # Training of the DNN 
    def train(self,batch_size, dnn_epoch):
        for k in range(dnn_epoch):
            minibatch = random.sample(self.replay_buffer, batch_size)
            minibatch = np.vstack(minibatch)
            minibatch = minibatch.reshape(batch_size,5)
            state = torch.FloatTensor(np.vstack(minibatch[:,0]))
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
            target_Q = reward.cpu().squeeze() + self.gamma * Q_next.cpu().max(1)[0].view(batch_size, 1).squeeze() * (
                1 - np.array([state[e].cpu().mean() == next_state[e].cpu().mean() for e in range(len(next_state))])
            ) 
            
            # Forward 
            # Loss calculation based on loss function
            target_Q = target_Q.float()
            Q_main = self.main_network(state).gather(1, action).squeeze()
            loss = self.loss_func(target_Q.cpu().detach(), Q_main.cpu())
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            # For gradient clipping
            for param in self.main_network.parameters():
                param.grad.data.clamp_(-1,1)
            # Gradient descent
            self.optimizer.step()


u_env = UAVenv()
GRID_SIZE = u_env.GRID_SIZE
NUM_UAV = u_env.NUM_UAV
NUM_USER = u_env.NUM_USER
num_episode = 201
num_epochs = 100
discount_factor = 0.95
alpha = 3.5e-4
batch_size = 512
update_rate = 10  #50
dnn_epoch = 1
epsilon = 0.90
epsilon_min = 0.10
epsilon_decay = 400
random.seed(10)

# Keeping track of the episode reward
episode_reward = np.zeros(num_episode)
episode_user_connected = np.zeros(num_episode)

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
        states_ten = torch.from_numpy(states)
        for k in range(NUM_UAV):
            state = states_ten[k, :]
            action = UAV_OB[k].epsilon_greedy(state.float())
            drone_act_list.append(action + 1)


        # Find the global reward for the combined set of actions for the UAV
        temp_data = u_env.step(drone_act_list)
        reward = temp_data[1]
        done = temp_data[2]
        next_state = u_env.get_state()

        # Store the transition information
        for k in range(NUM_UAV):
            ## Ommiting this in this run ##
            # Sharing of all the information of both state and reward between the UAVs 
            # Should not use collective number of user as reward but store the state and the reward of others in it's replay buffer too
            # Try this in a more complex setup with lower number of UAV and bigger grid size for better result
            # This in theory should perform far better
            state = states_ten[k, :]
            action = drone_act_list[k] - 1
            next_sta = next_state[k, :]
            reward_ind = reward[k]
            UAV_OB[k].store_transition(state, action, reward_ind, next_sta, done)

        
        episode_reward[i_episode] += sum(reward)
        episode_user_connected[i_episode] += temp_data[4]

        states = next_state


        for k in range(NUM_UAV):
            if len(UAV_OB[k].replay_buffer) > batch_size:
                UAV_OB[k].train(batch_size, dnn_epoch)

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
                best_next_action = torch.max(Q_values.cpu(), 1)[1].data.numpy()
                best_next_action = best_next_action[0]
                drone_act_list.append(best_next_action + 1)
            temp_data = u_env.step(drone_act_list)
            states = u_env.get_state()
            states_fin = states
            # print(temp_data)
            if best_result < temp_data[4]:
                best_result = temp_data[4]
                best_state = states
        u_env.render(ax1)
        plt.title("Intermediate state of UAV in this episode")
        print(drone_act_list)
        print("Number of user connected in ",i_episode," episode is: ", temp_data[4])

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
plt.plot(range(0, num_episode), episode_user_connected)
plt.xlabel("Episode")
plt.ylabel("Connected User in Episode")
plt.title("Episode vs Connected User in Epsisode")
plt.show()
fig = plt.figure()
smoothed = smooth(episode_reward, 10)
plt.plot(range(0, num_episode-10), smoothed[0:len(smoothed)-10] )
plt.xlabel("Episode")
plt.ylabel("Episodic Reward")
plt.title("Smoothed Epidode vs Episodic Reward")
plt.show()
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