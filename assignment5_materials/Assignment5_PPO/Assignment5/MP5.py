import sys
import gym
import torch
import pylab
import random
import numpy as np
import time
from collections import deque
from datetime import datetime
from copy import deepcopy
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from utils import *
from agent import *
from model import *
from config import *

env = gym.make('SpaceInvadersDeterministic-v4')
#env.render()

number_lives = find_max_lifes(env)
state_size = env.observation_space.shape
action_size = env.action_space.n
rewards, episodes = [], []

agent = Agent(action_size)

# Added to continue training

"""
agent.policy_net.load_state_dict(torch.load("./save_model/spaceinvaders_ppo"))
agent.update_target_net()
agent.policy_net.train()
"""

evaluation_reward = deque(maxlen=evaluation_reward_length)
frame = 0
memory_size = 0

for e in range(EPISODES):
    done = False
    score = 0

    history = np.zeros([5, 84, 84], dtype=np.uint8)
    step = 0
    d = False
    state = env.reset()
    life = number_lives

    get_init_state(history, state)

    while not done:
        step += 1
        frame += 1
        if render_breakout:
            env.render()

        # Select and perform an action
        curr_state = history[3,:,:]
        action, value = agent.get_action(np.float32(history[:4, :, :]) / 255.)

        
        next_state, reward, done, info = env.step(action)

        frame_next_state = get_frame(next_state)
        history[4, :, :] = frame_next_state
        terminal_state = check_live(life, info['ale.lives'])

        life = info['ale.lives']
        #r = np.clip(reward, -1, 1)
        r = reward
        """
        if terminal_state:
            r -= 20
        """
        # Store the transition in memory 
        
        agent.memory.push(deepcopy(curr_state), action, r, terminal_state, value, 0, 0)
        # Start training after random sample generation
        if(frame % train_frame == 0):
            _, frame_next_val = agent.get_action(np.float32(history[1:, :, :]) / 255.)
            agent.train_policy_net(frame, frame_next_val)
            # Update the target network
            agent.update_target_net()
        score += r
        history[:4, :, :] = history[1:, :, :]

        if frame % 50000 == 0:
            print('now time : ', datetime.now())
            rewards.append(np.mean(evaluation_reward))
            episodes.append(e)
            #pylab.plot(episodes, rewards, 'b')
            #pylab.savefig("./save_graph/spaceinvaders_ppo.png")
            torch.save(agent.policy_net.state_dict(), "./save_model/spaceinvaders_ppo")

        if done:
            evaluation_reward.append(score)
            # every episode, plot the play time
            print("episode:", e, "  score:", score, "  memory length:",
                  len(agent.memory), "  epsilon:", agent.epsilon, "   steps:", step,
                  "    evaluation reward:", np.mean(evaluation_reward))

            # if the mean of scores of last 10 episode is bigger than 400
            # stop training
            if np.mean(evaluation_reward) > 700 and len(evaluation_reward) > 40:
                torch.save(agent.policy_net.state_dict(), "./save_model/spaceinvaders_ppo")
                sys.exit()
                
torch.save(agent.policy_net.state_dict(), "./save_model/spaceinvaders_ppo")
