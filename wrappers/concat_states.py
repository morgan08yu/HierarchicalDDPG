import gym.spaces
import gym.wrappers
import numpy as np


def concat_states(state):
    history = state["history"]
    weights = state["weights"]
    weight_insert_shape = (history.shape[0], 1, history.shape[2])
    if len(weights) - 1 == history.shape[0]:
       weight_insert = np.ones(weight_insert_shape) * weights[1:, np.newaxis, np.newaxis]
    elif len(weights) - 1 == history.shape[2]:
       weight_insert = np.ones(weight_insert_shape) * weights[np.newaxis, np.newaxis, 1:]
    else:
       weight_insert = np.ones(weight_insert_shape) * weights[np.newaxis, 1:, np.newaxis]
    # weight_insert = np.ones(weight_insert_shape) * weights[np.newaxis, np.newaxis, :]  # TODO change here
    # weight_insert = np.ones(weight_insert_shape) * weights[np.newaxis, 1:, np.newaxis]
    state = np.concatenate([weight_insert, history], axis=1)
    return state

def concat(state):
    history = state["history"]
    weights1 = state["weights1"]
    # weights2 = state["weights2"]
    weight_insert_shape = (history.shape[0], 1, history.shape[2])
    if len(weights1) - 1 == history.shape[0]:
        weight_insert1 = np.ones(weight_insert_shape) * weights1[1:, np.newaxis, np.newaxis]
    elif len(weights1) - 1 == history.shape[2]:
        weight_insert1 = np.ones(weight_insert_shape) * weights1[np.newaxis, np.newaxis, 1:]
    else:
        weight_insert1 = np.ones(weight_insert_shape) * weights1[np.newaxis, 1:, np.newaxis]
    # if len(weights2) - 1 == history.shape[0]:
    #    weight_insert2 = np.ones(weight_insert_shape) * weights1[1:, np.newaxis, np.newaxis]
    # elif len(weights2) - 1 == history.shape[2]:
    #    weight_insert2 = np.ones(weight_insert_shape) * weights1[np.newaxis, np.newaxis, 1:]
    # else:
    #    weight_insert2 = np.ones(weight_insert_shape) * weights1[np.newaxis, 1:, np.newaxis]
    state = np.concatenate([weight_insert1, history], axis=1) # TODO remove 2nd weight
    return state

class ConcatStates(gym.Wrapper):
    """
    Concat both state arrays for models that take a single inputs.

    Usage:
        env = ConcatStates(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md
    """

    def __init__(self, env):
        super().__init__(env)
        hist_space = self.observation_space.spaces["history"]
        hist_shape = hist_space.shape
        self.observation_space = gym.spaces.Box(-np.inf, +np.inf, shape=(
            hist_shape[0], hist_shape[1] + 1, hist_shape[2]))

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # concat the two state arrays, since some models only take a single output
        state = concat_states(state)
        return state, reward, done, info

    def reset(self):
        state = self.env.reset()
        return concat_states(state)


class RobConcatStates(gym.Wrapper):
    """
    Concat both state arrays for models that take a single inputs.

    Usage:
        env = ConcatStates(env)

    Ref: https://github.com/openai/gym/blob/master/gym/wrappers/README.md
    """

    def __init__(self, env):
        super().__init__(env)
        hist_space = self.observation_space.spaces["history"]
        hist_shape = hist_space.shape
        self.observation_space = gym.spaces.Box(-np.inf, +np.inf, shape=(
            hist_shape[0], hist_shape[1] + 1, hist_shape[2]))
#TODO remove 2nd weights
    def step(self, action1, action2):

        state, reward_gt, reward_at, done, info, z = self.env.step(action1, action2)

        # concat the two state arrays, since some models only take a single output
        state = concat(state)

        return state, reward_gt, reward_at, done, info, z

    def reset(self):
        state = self.env.reset()
        return concat(state)