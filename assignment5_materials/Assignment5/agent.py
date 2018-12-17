import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory
from model import *
from utils import *
from config import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, action_size):
        self.load_model = False

        self.action_size = action_size
        self.loss = nn.SmoothL1Loss()

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)
        self.target_net = DQN(action_size)
        self.target_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

        # initialize target net
        self.update_target_net()

        if self.load_model:
            self.policy_net = torch.load('save_model/breakout_dqn')

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using epsilon-greedy policy"""
    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE ####
            # Choose a random action
            return random.randint(0, self.action_size-1)
        else:
            ### CODE ####
            # 
            state = torch.from_numpy(state).to(device).unsqueeze(0)
            vals = self.policy_net(state)
            maxQ, a = torch.max(vals, dim=1)
            return a

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        actions = np.array(list(mini_batch[1]))
        rewards = np.array(list(mini_batch[2]))
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] # checks if the game is over
        
        ### Converts everything to tensor
        states = torch.from_numpy(states).to(device)
        actions = torch.from_numpy(actions).to(device)
        rewards = torch.from_numpy(rewards).float().to(device)
        next_states = torch.from_numpy(next_states).to(device)
        dones = torch.from_numpy(np.uint8(dones)).to(device)
        
        # Compute Q(s_t, a) - Q of the current state
        ### CODE ####
        n = states.shape[0]
        actions = actions.reshape((n, 1))
        Q_curr = self.policy_net(states).gather(1, actions).reshape((n,))
        
        # Compute Q function of next state
        ### CODE ####
        nonterminals = 1-dones
        next_in = next_states[nonterminals]
        targ_next = self.target_net(next_in)
        
        
        # Find maximum Q-value of action at next state from target net
        ### CODE ####
        Q_next = Variable(torch.zeros(len(states))).to(device)
        Q_next[nonterminals], _ = torch.max(targ_next, dim=1)
        Q_next = rewards + self.discount_factor * Q_next
        
        # Compute the Huber Loss
        ### CODE ####
        loss = self.loss(Q_curr, Q_next.detach())
        
        # Optimize the model 
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        
        ### print values
        #print(Q_curr.data[0].cpu().numpy(), Q_next.data[0].cpu().numpy())
        
