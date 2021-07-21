import gym.wrappers
import os
os.sys.path.append(os.path.abspath('.'))
#os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system'))
from utils.util import softmax
import pandas as pd
import numpy as np
from scipy.special import logit

eps = 1e-12
def softmax(w):
    """softmax implemented in numpy."""
    log_eps = np.log(eps)
    w = np.clip(w, log_eps, -log_eps)  # avoid inf/nan
    e = np.exp(np.array(w))
    dist = e / np.sum(e)
    return dist

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logit(x):
    return np.log(x/(1-x))

class SoftmaxActions(gym.Wrapper):
    """
    Environment wrapper to softmax actions.

    Usage:
        env = gym.make('Pong-v0')
        env = SoftmaxActions(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md

    """

    def step(self, action):
        # also it puts it in a list
        if isinstance(action, list):
            action = action[0]

        if isinstance(action, dict):
            action = list(action[k] for k in sorted(action.keys()))
        #action = sigmoid(action)
        act = softmax(action)
        #action = logit(action)
        return self.env.step(act)


class RobSoftmaxActions(gym.Wrapper):
    """
    Environment wrapper to softmax actions.

    Usage:
        env = gym.make('Pong-v0')
        env = SoftmaxActions(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md

    """

    def step(self, action1, action2):
        # also it puts it in a list
        if isinstance(action1, list):
            action1 = action1[0]
        if isinstance(action2, list):
            action2 = action2[0]
        if isinstance(action1, dict):
            action1 = list(action1[k] for k in sorted(action1.keys()))
        if isinstance(action2, dict):
            action2 = list(action2[k] for k in sorted(action2.keys()))
        #action = sigmoid(action)
        action1 = softmax(action1)
        action2 = softmax(action2)
        #action = logit(action)
        #action = np.sin(action*3.14/2)
        return self.env.step(action1, action2)