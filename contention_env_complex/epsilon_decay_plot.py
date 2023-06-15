from ast import Num
import random
import numpy as np
import math
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
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
from scipy.io import savemat

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
PATH_DATASETS = os.environ.get("PATH_DATASETS", ".")

SEED = 1
torch.manual_seed(SEED)
np.random.seed(SEED)

## GPU configuration use for faster processing
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = "cpu"



global steps_done
steps_done = 0
num_episode = 351
num_epochs = 100
discount_factor = 0.95
alpha = 1e-4
batch_size = 512
update_rate = 10  #50
dnn_epoch = 1
epsilon = 0.9
epsilon_min = 0.1
epsilon_decay = 1500
random.seed(SEED)

# Deployment of epsilon greedy policy
def epsilon_greedy( epsilon_min, epsilon, epsilon_decay, steps_done):
    # Epsilon decay policy is employed for faster convergence
    epsilon_thres = epsilon_min + (epsilon - epsilon_min) * math.exp(-1*steps_done/epsilon_decay)
    
    return epsilon_thres

# Keeping track of the episode reward
episode_reward = np.zeros(num_episode)
episode_user_connected = np.zeros(num_episode)
steps_done = 0
thes_append = np.array(range(num_episode), dtype=np.float32)
fig = plt.figure()
for i_episode in range(num_episode):
    print(i_episode)
    
    for t in range(num_epochs):
        thres = epsilon_greedy(epsilon_min, epsilon, epsilon_decay, steps_done)
        thes_append[i_episode] = thres
        print(thres)
        steps_done += 1 
print(thes_append)
plt.plot(range(num_episode), thes_append)
plt.xlabel('Episodes')
plt.ylabel('Epsilon')
plt.title('Episode vs epsilon decay')
plt.pause(50)
fig.show()


