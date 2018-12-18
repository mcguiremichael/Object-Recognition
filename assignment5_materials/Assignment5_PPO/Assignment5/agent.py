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
        self.lam = 0.995
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.eps_denom = 1e-2
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000
        self.c1 = 1.0       # Weight for value loss
        self.c2 = 0.01      # Weight for entropy loss
        self.num_epochs = 3

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net and the target net
        self.policy_net = PPO(action_size)
        self.policy_net.to(device)
        self.target_net = PPO(action_size)
        self.target_net.to(device)

        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

        # initialize target net
        self.update_target_net()

        if self.load_model:
            self.policy_net = torch.load('save_model/breakout_dqn')

    # after some time interval update the target net to be same with policy net
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using action probabilities"""
    def get_action(self, state):
        state = torch.from_numpy(state).to(device).unsqueeze(0)
        probs, val = self.policy_net(state)
        probs = probs.detach().cpu().numpy()[0]
        val = val.detach().cpu().numpy()
        action = self.select_action(probs)
        return action, val
        
    def select_action(self, probs):
        candidate = random.random()
        total = probs[0]
        i = 0
        while (total < candidate and total < 1.0):
            i += 1
            total += probs[i]

        return i
        
    def entropy(self, probs):
        return -(torch.sum(probs * torch.log(probs), 1)).mean()
        
    def train_policy_net(self, frame):
    
        print("Training network")
    
        # Memory computes targets for value network, and advantag es for policy iteration
        self.memory.compute_vtargets_adv(self.discount_factor, self.lam)
        
        # Should be integer. len(self.memory) should be a multiple of batch_size.
        num_iters = int(len(self.memory) / batch_size)
        
        
        
        for i in range(self.num_epochs):
            print("Iteration %d" % (i+1))
            
            pol_loss = 0.0
            vf_loss = 0.0
            ent_total = 0.0
        
            for i in range(num_iters):
                
                mini_batch = self.memory.sample_mini_batch(frame)
                mini_batch = np.array(mini_batch).transpose()
                
                history = np.stack(mini_batch[0], axis=0)
                states = np.float32(history[:, :4, :, :]) / 255.
                actions = np.array(list(mini_batch[1]))
                rewards = np.array(list(mini_batch[2]))
                next_states = np.float32(history[:, 1:, :, :]) / 255.
                dones = mini_batch[3]
                v_returns = mini_batch[5].astype(np.float32)
                advantages = mini_batch[6].astype(np.float32)
                
                
                
                
                ### Converts everything to tensor
                states = torch.from_numpy(states).to(device)
                actions = torch.from_numpy(actions).to(device)
                rewards = torch.from_numpy(rewards).to(device)
                next_states = torch.from_numpy(next_states).to(device)
                dones = torch.from_numpy(np.uint8(dones)).to(device)
                v_returns = torch.from_numpy(v_returns).to(device)
                advantages = torch.from_numpy(advantages).to(device)
                
                # Normalize advantages
                advantages = (advantages - advantages.mean()) / (advantages.std() + self.eps_denom)
                
                
                
                ### Get probs, val from current state for curr, old policies
                n = states.shape[0]
                actions = actions.reshape((n, 1))
                
                
                curr_probs, curr_vals = self.policy_net(states)
                old_probs, old_vals = self.target_net(states)
                
                
                curr_prob_select = curr_probs.gather(1, actions).reshape((n,))
                old_prob_select = old_probs.gather(1, actions).reshape((n,))
                
                
                
                ### Compute ratios
                ratio = torch.exp(torch.log(curr_prob_select) - torch.log(old_prob_select.detach() + self.eps_denom))
                ratio_adv = ratio * advantages
                bounded_adv = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * advantages
                
                
                      
                pol_avg = -torch.min(ratio_adv, bounded_adv).mean()

                """
                if (i == 0):
                    print(ratio, ratio_adv, bounded_adv, pol_avg)  
                """
                ### Compute value and loss
                value_loss = self.loss(curr_vals, v_returns.detach())
                
                ent = self.entropy(curr_probs)

                total_loss = pol_avg + self.c1 * value_loss - self.c2 * ent
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.clip_gradients(0.1)
                self.optimizer.step()
                
                pol_loss += pol_avg.detach().cpu()[0]
                vf_loss += value_loss.detach().cpu()[0]
                ent_total += ent.detach().cpu()[0]
                
            pol_loss /= num_iters
            vf_loss /= num_iters
            ent_total /= num_iters
            print("Policy loss: %f. Value loss: %f. Entropy: %f." % (pol_loss, vf_loss, ent_total))
        
    def clip_gradients(self, clip):
        
        ### Clip the gradients of self.policy_net
        for param in self.policy_net.parameters():
            if param.grad is None:
                continue
            param.grad.data = param.grad.data.clamp(-clip, clip)
        

































        
