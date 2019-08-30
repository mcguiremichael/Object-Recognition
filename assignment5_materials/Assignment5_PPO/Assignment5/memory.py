from config import *
from collections import deque
import numpy as np
import random


class ReplayMemory(object):
    def __init__(self, mode='PPO'):
        self.num_agents = num_envs
        assert(Memory_capacity % self.num_agents == 0)
        self.agent_mem_size = int(Memory_capacity / self.num_agents)
        self.memory = [deque(maxlen=self.agent_mem_size) for i in range(self.num_agents)]
        self.access_num = 0
        self.indices = []
        self.update_indices()
        self.train_len = len(self.indices)
        self.reset_num = int(self.train_len / batch_size)
        self.mode = mode
    
    def push(self, index, history, action, reward, done, vtarg, ret, adv):
        # history, action, reward, done, vtarg, adv
        hidden = None
        if self.mode == 'PPO_LSTM':
            [history, hidden] = history
        self.memory[index].append([history, action, reward, done, vtarg, ret, adv, hidden])
        
    def update_indices(self):
        self.indices = []
        for i in range(self.num_agents):
            lower = i*self.agent_mem_size
            upper = (i+1)*self.agent_mem_size - (HISTORY_SIZE+1)
            self.indices += list(range(lower, upper))
        random.shuffle(self.indices)

    def sample_mini_batch(self, frame):
        
        
        
        
        mini_batch = []
        if frame >= len(self.indices):
            sample_range = len(self.indices)
        else:
            sample_range = frame

        depth = HISTORY_SIZE-1
            

        # history size
        #sample_range -= (HISTORY_SIZE + 1)
        
        lower = batch_size*self.access_num
        upper = min((batch_size*(self.access_num+1)), sample_range)

        idx_sample = self.indices[lower:upper]
        for i in idx_sample:
            sample = []
            
            env_idx = int(i / self.agent_mem_size)
            frame_idx = i % self.agent_mem_size
            
            for j in range(HISTORY_SIZE + 1):
                sample.append(self.memory[env_idx][frame_idx+j])

            sample = np.array(sample)
            mini_batch.append([np.stack(sample[:, 0], axis=0), sample[depth, 1], sample[depth, 2], sample[depth, 3], sample[depth, 4], sample[depth, 5], sample[depth, 6], sample[0, 7]])

        self.access_num = (self.access_num + 1) % self.reset_num
        if (self.access_num == 0):
            self.update_indices()


        return mini_batch
        
        
    def compute_vtargets_adv(self, gamma, lam, frame_next_val):
        for i in range(self.num_agents):
            mem = self.memory[i]
            N = len(mem)
            prev_gae_t = 0
            
            for j in reversed(range(N)):
                
                if j+1 == N:
                    vnext = frame_next_val[i]
                    nonterminal = 1
                else:
                    vnext = mem[j+1][4]
                    nonterminal = 1 - mem[j+1][3]   # 1 - done
                delta = mem[j][2] + gamma * vnext * nonterminal - mem[j][4]
                gae_t = delta + gamma * lam * nonterminal * prev_gae_t
                mem[j][6] = gae_t     # advantage
                mem[j][5] = gae_t + mem[j][4]  # advantage + value
                prev_gae_t = gae_t
        """
        for i in range(self.num_agents):
            mem = self.memory[i]
            for j in range(len(mem)):
                print(mem[j][1:])
        """
        
        
        """
        N = len(self)
        
        prev_gae_t = 0
       
        
        for i in reversed(range(N)):
            
            if i+1 == N:
                vnext = frame_next_val
                nonterminal = 1
            else:
                vnext = self.memory[i+1][4]
                nonterminal = 1 - self.memory[i+1][3]    # 1 - done
            delta = self.memory[i][2] + gamma * vnext * nonterminal - self.memory[i][4]
            gae_t = delta + gamma * lam * nonterminal * prev_gae_t
            self.memory[i][6] = gae_t    # advantage
            self.memory[i][5] = gae_t + self.memory[i][4]  # advantage + value
            prev_gae_t = gae_t
        
        """

    def __len__(self):
        return sum([len(self.memory[i]) for i in range(len(self.memory))])
