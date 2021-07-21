import gym.wrappers
import os
os.sys.path.append(os.path.abspath('.'))
#os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system'))
from utils.util import softmax
import pandas as pd
import numpy as np
eps=1e-7
from scipy.special import logit

class LogitActions(gym.Wrapper):
    def step(self, action):
        # also it puts it in a list
        if isinstance(action, list):
            action = action[0]

        if isinstance(action, dict):
            action = list(action[k] for k in sorted(action.keys()))
        action =np.clip(action, eps, 1-eps)
        action = logit(action)
        return self.env.step(action)