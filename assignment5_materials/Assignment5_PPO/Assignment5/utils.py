import numpy as np
from skimage.transform import resize
from skimage.color import rgb2gray
from config import *
import gym
import random

def find_max_lifes(env):
    env.reset()
    _, _, _, info = env.step(0)
    return info['ale.lives']

def check_live(life, cur_life):
    if life > cur_life:
        return True
    else:
        return False

def get_frame(X):
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x

def get_init_state(history, s):
    for i in range(HISTORY_SIZE):
        history[i, :, :] = get_frame(s)
        
def get_score_range(name):
    env = gym.make(name)
    n = env.action_space.n
    max_score = -np.inf
    min_score = np.inf
    for i in range(100):
        done = False
        env.reset()
        while not done:
            action = random.randint(0, n-1)
            _, reward, done, _ = env.step(action)
            if (reward > max_score):
                max_score = reward
            if (reward < min_score):
                min_score = reward
    return [min_score, max_score]

        
