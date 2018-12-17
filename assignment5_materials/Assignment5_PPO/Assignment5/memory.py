from config import *
from collections import deque
import numpy as np
import random


class ReplayMemory(object):
    def __init__(self):
        self.memory = deque(maxlen=Memory_capacity)
    
    def push(self, history, action, reward, done, vtarg, ret, adv):
        # history, action, reward, done, vtarg, adv
        self.memory.append([history, action, reward, done, vtarg, ret, adv])

    def sample_mini_batch(self, frame):
        mini_batch = []
        if frame >= Memory_capacity:
            sample_range = Memory_capacity
        else:
            sample_range = frame

        # history size
        sample_range -= (HISTORY_SIZE + 1)

        idx_sample = random.sample(range(sample_range), batch_size)
        for i in idx_sample:
            sample = []
            for j in range(HISTORY_SIZE + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample)
            mini_batch.append([np.stack(sample[:, 0], axis=0), sample[3, 1], sample[3, 2], sample[3, 3], sample[3, 4], sample[3, 5], sample[3, 6]])

        return mini_batch
        
        
    def compute_vtargets_adv(self, gamma, lam):
        N = len(self)
        
        prev_gae_t = 0
        
        for i in reversed(range(N)):
            
            if i+1 == N:
                vnext = 0
                nonterminal = 0
            else:
                vnext = self.memory[i+1][4]
                nonterminal = 1 - self.memory[i+1][3]    # 1 - done
            delta = self.memory[i][2] + gamma * vnext * nonterminal - self.memory[i][4]
            gae_t = delta + gamma * lam * nonterminal * prev_gae_t
            self.memory[i][6] = gae_t    # advantage
            self.memory[i][5] = gae_t + self.memory[i][4]  # advantage + value
            prev_gae_t = gae_t
            
            


    def __len__(self):
        return len(self.memory)
