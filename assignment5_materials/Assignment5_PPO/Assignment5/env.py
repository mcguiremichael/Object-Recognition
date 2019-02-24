import gym
import numpy as np
from utils import *
from config import *


class GameEnv():

    def __init__(self, name):
        self._env = gym.make(name)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        
        self.state = self._env.reset()
        self.reward = 0
        self.done = False
        self.info = None
        
        self.history = np.zeros([HISTORY_SIZE+1,84,84], dtype=np.uint8)
        self.number_lives = find_max_lifes(self._env)
        
        self.score = 0
        self.life = find_max_lifes(self._env)
        get_init_state(self.history, self.state)
        
        
    def step(self, action):
        return self._env.step(action)
        
    def reset(self):
        return self._env.reset()
        
    def render(self):
        self._env.render()
