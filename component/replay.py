
import numpy as np
import torch
import random
import torch.multiprocessing as mp

class Replay:
    def __init__(self, memory_size, batch_size, drop_prob=0, to_np=True):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.data = []
        self.pos = 0
        self.drop_prob = drop_prob
        self.to_np = to_np

    def feed(self, experience):
        if np.random.rand() < self.drop_prob:
            return
        if self.pos >= len(self.data):
            self.data.append(experience)
        else:
            self.data[self.pos] = experience
        self.pos = (self.pos + 1) % self.memory_size

    def feed_batch(self, experience):
        for exp in experience:
            self.feed(exp)

    def sample(self, batch_size=None):
        if self.empty():
            return None
        if batch_size is None:
            batch_size = self.batch_size

        sampled_indices = [np.random.randint(0, len(self.data)) for _ in range(batch_size)]
        sampled_data = [self.data[ind] for ind in sampled_indices]
        sampled_data = zip(*sampled_data)
        if self.to_np:
            sampled_data = list(map(lambda x: np.asarray(x), sampled_data))
        return sampled_data

    def size(self):
        return len(self.data)

    def empty(self):
        return not len(self.data)

    def shuffle(self):
        np.random.shuffle(self.data)

    def clear(self):
        self.data = []
        self.pos = 0

class Replay2:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype

        self.states = None
        self.actions = np.empty(self.memory_size, dtype=np.int8)
        self.rewards = np.empty(self.memory_size)
        self.next_states = None
        self.terminals = np.empty(self.memory_size, dtype=np.int8)

        self.pos = 0
        self.full = False


    def feed(self, experience):
        state, action, reward, next_state, done = experience

        if self.states is None:
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos][:] = next_state
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def sample(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]

class HybridRewardReplay:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype

        self.states = None
        self.actions = np.empty(self.memory_size, dtype=np.int8)
        self.rewards = None
        self.next_states = None
        self.terminals = np.empty(self.memory_size, dtype=np.int8)

        self.pos = 0
        self.full = False


    def feed(self, experience):
        state, action, reward, next_state, done = experience

        if self.states is None:
            self.rewards = np.empty((self.memory_size, ) + reward.shape, dtype=self.dtype)
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos] = action
        self.rewards[self.pos][:] = reward
        self.next_states[self.pos][:] = next_state
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def sample(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]

class SharedReplay:
    def __init__(self, memory_size, batch_size, state_shape, action_shape):
        self.memory_size = memory_size
        self.batch_size = batch_size

        self.states = torch.zeros((self.memory_size, ) + state_shape)
        self.actions = torch.zeros((self.memory_size, ) + action_shape)
        self.rewards = torch.zeros(self.memory_size)
        self.next_states = torch.zeros((self.memory_size, ) + state_shape)
        self.terminals = torch.zeros(self.memory_size)

        self.states.share_memory_()
        self.actions.share_memory_()
        self.rewards.share_memory_()
        self.next_states.share_memory_()
        self.terminals.share_memory_()

        self.pos = 0
        self.full = False
        self.buffer_lock = mp.Lock()

    def feed_(self, experience):
        state, action, reward, next_state, done = experience
        self.states[self.pos][:] = torch.FloatTensor(state)
        self.actions[self.pos][:] = torch.FloatTensor(action)
        self.rewards[self.pos] = reward
        self.next_states[self.pos][:] = torch.FloatTensor(next_state)
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def sample_(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = torch.LongTensor(np.random.randint(0, upper_bound, size=self.batch_size))
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]

    def feed(self, experience):
        with self.buffer_lock:
            self.feed_(experience)

    def sample(self):
        with self.buffer_lock:
            return self.sample_()

class HighDimActionReplay:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype

        self.states = None
        self.actions = None
        self.rewards = np.empty(self.memory_size)
        self.next_states = None
        self.terminals = np.empty(self.memory_size, dtype=np.int8)

        self.pos = 0
        self.full = False


    def feed(self, experience):
        state, action, reward, next_state, done = experience

        if self.states is None:
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.actions = np.empty((self.memory_size, ) + action.shape)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos][:] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos][:] = next_state
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def sample(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.terminals[sampled_indices]]


class RobHighDimActionReplay:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.states = None
        self.action_gt = None
        self.action_at = None
        self.rewards = np.empty(self.memory_size)
        #self.rewards_best = np.empty(self.memory_size)
        self.next_states = None
        #self.window_worst = np.empty(self.memory_size)
        #self.window_best = np.empty(self.memory_size)
        self.terminals = np.empty(self.memory_size, dtype=np.int8)
        self.pos = 0
        self.full = False


    def feed(self, experience):
        state, action_gt, action_at, reward, next_state, done = experience
        if self.states is None:
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.actions_gt = np.empty((self.memory_size, ) + action_gt.shape)
            self.actions_at = np.empty((self.memory_size, ) + action_at.shape)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions_gt[self.pos][:] = action_gt
        self.actions_at[self.pos][:] = action_at
        #self.rewards_best[self.pos] = reward_best
        self.rewards[self.pos] = reward
        #self.window_best[self.pos] = window_best
        #self.window_worst[self.pos] = window_worst
        #self.window_best[self.pos] = window_best
        self.next_states[self.pos][:] = next_state
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def sample(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        return [self.states[sampled_indices],
                self.actions_gt[sampled_indices],
                self.actions_at[sampled_indices],
                #self.rewards_best[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                #self.window_worst[sampled_indices],
                #self.window_best[sampled_indices],
                self.terminals[sampled_indices]]

class GeneralReplay:
    def __init__(self, memory_size, batch_size):
        self.buffer = []
        self.memory_size = memory_size
        self.batch_size = batch_size

    def feed(self, experiences):
        for experience in zip(*experiences):
            self.buffer.append(experience)
            if len(self.buffer) > self.memory_size:
                del self.buffer[0]

    def sample(self):
        sampled = zip(*random.sample(self.buffer, self.batch_size))
        return sampled

    def clear(self):
        self.buffer = []

    def full(self):
        return len(self.buffer) == self.memory_size


from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, memory_size, batch_size):
        """
        The right side of the deque contains the most recent experiences
        """
        self.buffer_size = memory_size
        self.count = 0
        self.batch_size = batch_size
        self.buffer = deque()
        random.seed(123)

    def feed(self, experience):
        state, action, reward, next_state, done = experience
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample(self):
        if self.count < self.batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, self.batch_size)

        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        s2_batch = np.array([_[3] for _ in batch])
        t_batch = np.array([_[4] for _ in batch])
        return s_batch, a_batch, r_batch, s2_batch, t_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0

class ReplayAlpha:
    def __init__(self, memory_size, batch_size, dtype=np.float32):
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.dtype = dtype
        self.alpha = np.empty(self.memory_size)
        self.states = None
        self.actions = None
        self.rewards = np.empty(self.memory_size)
        self.next_states = None
        self.terminals = np.empty(self.memory_size, dtype=np.int8)
        self.pos = 0
        self.full = False

    def feed(self, experience):
        state, action, reward, next_state, alpha, done = experience

        if self.states is None:
            self.states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)
            self.actions = np.empty((self.memory_size, ) + action.shape)
            self.next_states = np.empty((self.memory_size, ) + state.shape, dtype=self.dtype)

        self.states[self.pos][:] = state
        self.actions[self.pos][:] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos][:] = next_state
        self.alpha[self.pos] = alpha
        self.terminals[self.pos] = done

        self.pos += 1
        if self.pos == self.memory_size:
            self.full = True
            self.pos = 0

    def size(self):
        if self.full:
            return self.memory_size
        return self.pos

    def sample(self):
        upper_bound = self.memory_size if self.full else self.pos
        sampled_indices = np.random.randint(0, upper_bound, size=self.batch_size)
        return [self.states[sampled_indices],
                self.actions[sampled_indices],
                self.rewards[sampled_indices],
                self.next_states[sampled_indices],
                self.alpha[sampled_indices],
                self.terminals[sampled_indices]]