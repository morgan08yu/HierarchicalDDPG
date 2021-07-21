import numpy as np
from collections import namedtuple
import random

'''
class EXP :
	def __init__(self, s, a, s1, r, done) :
		self.s = s
		self.a = a
		self.s1 = s1
		self.r = r
		self.done = done
'''
EXP = namedtuple('EXP', ('state', 'action', 'next_state', 'reward', 'done'))
#Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))
TransitionPR = namedtuple('TransitionPR', ('idx', 'priority', 'state', 'action', 'next_state', 'reward', 'done'))


class EXP_RNN(EXP):
    def __init__(self, s, a, s1, r, done, cnn_states):
        EXP.__init__(self, s=s, a=a, s1=s1, r=r, done=done)
        self.cnn_states = cnn_states

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.length = 0
        self.counter = 0
        self.alpha = alpha
        self.epsilon = 1e-6
        self.capacity = int(capacity)
        self.tree = np.zeros(2 * self.capacity - 1)
        self.data = np.zeros(self.capacity, dtype=object)

    def reset(self):
        self.__init__(capacity=self.capacity, alpha=self.alpha)

    def add(self, exp, priority):
        idx = self.counter + self.capacity - 1

        self.data[self.counter] = exp

        self.counter += 1
        self.length = min(self.length + 1, self.capacity)
        if self.counter >= self.capacity:
            self.counter = 0

        self.update(idx, priority)

    def priority(self, error):
        return (error + self.epsilon) ** self.alpha

    def update(self, idx, priority):
        change = priority - self.tree[idx]

        self.tree[idx] = priority

        self._propagate(idx, change)

    def _propagate(self, idx, change):
        parentidx = (idx - 1) // 2

        self.tree[parentidx] += change

        if parentidx != 0:
            self._propagate(parentidx, change)

    def __call__(self, s):
        idx = self._retrieve(0, s)
        dataidx = idx - self.capacity + 1
        data = self.data[dataidx]
        priority = self.tree[idx]

        return (idx, priority, data)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataidx = idx - self.capacity + 1

        data = self.data[dataidx]
        if not isinstance(data, EXP):
            raise TypeError

        priority = self.tree[idx]

        return (idx, priority, *data)

    def get_buffer(self):
        return [self.data[i] for i in range(self.capacity) if isinstance(self.data[i], EXP)]

    def _retrieve(self, idx, s):
        leftidx = 2 * idx + 1
        rightidx = leftidx + 1

        if leftidx >= len(self.tree):
            return idx

        if s <= self.tree[leftidx]:
            return self._retrieve(leftidx, s)
        else:
            return self._retrieve(rightidx, s - self.tree[leftidx])

    def total(self):
        return self.tree[0]

    def __len__(self):
        return self.length


class ReplayMem(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def feed(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

from collections import namedtuple, deque
from torch.utils.data import Dataset
import random
import numpy as np

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


# Segment tree data structure where parent node values are sum/max of children node values
class SegmentTree():
  def __init__(self, size):
    self.index = 0
    self.size = size
    self.full = False  # Used to track actual capacity
    self.sum_tree = [0] * (2 * size - 1)  # Initialise fixed size tree with all (priority) zeros
    self.data = [None] * size  # Wrap-around cyclic buffer
    self.max = 1  # Initial max value to return (1 = 1^ω)

  # Propagates value up tree given a tree index
  def _propagate(self, index, value):
    parent = (index - 1) // 2
    left, right = 2 * parent + 1, 2 * parent + 2
    self.sum_tree[parent] = self.sum_tree[left] + self.sum_tree[right]
    if parent != 0:
      self._propagate(parent, value)

  # Updates value given a tree index
  def update(self, index, value):
    self.sum_tree[index] = value  # Set new value
    self._propagate(index, value)  # Propagate value
    self.max = max(value, self.max)

  def append(self, data, value):
    self.data[self.index] = data  # Store data in underlying data structure
    self.update(self.index + self.size - 1, value)  # Update tree
    self.index = (self.index + 1) % self.size  # Update index
    self.full = self.full or self.index == 0  # Save when capacity reached
    self.max = max(value, self.max)

  # Searches for the location of a value in sum tree
  def _retrieve(self, index, value):
    left, right = 2 * index + 1, 2 * index + 2
    if left >= len(self.sum_tree):
      return index
    elif value <= self.sum_tree[left]:
      return self._retrieve(left, value)
    else:
      return self._retrieve(right, value - self.sum_tree[left])

  # Searches for a value in sum tree and returns value, data index and tree index
  def find(self, value):
    index = self._retrieve(0, value)  # Search for index of item from root
    data_index = index - self.size + 1
    return (self.sum_tree[index], data_index, index)  # Return value, data index, tree index

  # Returns data given a data index
  def get(self, data_index):
    return self.data[data_index % self.size]

  def total(self):
    return self.sum_tree[0]


class ReplayBufferDataset(Dataset):
    """
    Dataset implementation of the experience replay
    This class helps in the case of using multi gpu training
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)

    def __getitem__(self, index):
        return self.memory[index]


class ReplayBuffer_1(object):

    def __init__(self, capacity, seed,
                 priority_weight=None, priority_exponent=None,
                 priotirized_experience=False):
        self.capacity = capacity
        self.position = 0
        self.prioritize = priotirized_experience
        self.priority_weight = priority_weight  # Initial importance sampling weight β, annealed to 1 over course of training
        self.priority_exponent = priority_exponent
        if self.prioritize:
            self.memory = []
        else:
            self.memory = []
        # Seed for reproducible results
        np.random.seed(seed)

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample_batch(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def get_buffer_size(self):
        return len(self.memory)


# Use this replay buffer for non goal environments
class ReplayBufferDeque(object):

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


# Long term memory that uses resevoir sampling for adding items
class SelectiveExperienceReplayBuffer(object):
    pass
