import os
os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system/'))
from matplotlib import pyplot as plt
import matplotlib
#matplotlib.use('nbAgg', force=True)
matplotlib.rc('figure', figsize=[15, 10])
#matplotlib.use('TkAgg')
import seaborn as sns
import numpy as np
import threading
from numpy import random
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from collections import Counter
import tempfile
import logging
import time
import datetime
logger = log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()
log.info('%s logger started', __name__)
#from utils.data import read_stock_history, index_to_date, date_to_index, normalize
import matplotlib
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from collections import Counter
import tempfile
import logging
import time
import seaborn as sns
window = 30

steps = 128
import datetime
ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
save_path = '/Users/Morgans/Desktop/trading_system/log_TEST/test'
save_path
try:
    os.makedirs(os.path.dirname(save_path))
except OSError:
    pass

from tensorboard_logger import configure, log_value

tag = 'ddpg-' + ts
print('tensorboard --logdir ' + "runs/" + tag)
try:
    configure("runs/" + tag)
except ValueError as e:
    print(e)
    pass

from Environment.DDPGPEnv import PortfolioEnv
from utils.util import MDD, sharpe, softmax
from wrappers import SoftmaxActions, TransposeHistory, ConcatStates

from wrappers.logit import LogitActions
df_train = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/sh_data_set/poloniex_vol_sh.hf', key='train')
df_test = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/sh_data_set/poloniex_vol_sh.hf', key='test')

import gym

#from utils.normalizer import Normalizer
class Normalizer(object):
    def __init__(self, size, eps=1e-5, default_clip_range=np.inf):
        """A normalizer that ensures that observations are approximately distributed according to
        a standard Normal distribution (i.e. have mean zero and variance one).
        Args:
            size (int): the size of the observation to be normalized
            eps (float): a small constant that avoids underflows
            default_clip_range (float): normalized observations are clipped to be in
                [-default_clip_range, default_clip_range]
        """
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range  # always 5
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        self.lock = threading.Lock()
        self.running_mean = np.zeros(self.size, dtype=np.float32)
        self.running_std = np.ones(self.size, dtype=np.float32)
        self.running_sum = np.zeros(self.size, dtype=np.float32)
        self.running_sum_sq = np.zeros(self.size, dtype=np.float32)
        self.running_count = 1

    def update(self, v):
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    def normalize(self, v):
        # hard-coded to 5
        clip_range = self.default_clip_range
        return np.clip((v - self.running_mean) / self.running_std, -clip_range, clip_range).astype(np.float32)

    def recompute_stats(self):
        with self.lock:
            self.running_count += self.local_count
            self.running_sum += self.local_sum
            self.running_sum_sq += self.local_sumsq

            # Reset.
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0

        self.running_mean = self.running_sum / self.running_count
        self.running_std = np.sqrt(np.maximum(np.square(self.eps), self.running_sum_sq / self.running_count - np.square(self.running_sum/self.running_count)))


null_normaliser = lambda x: x

class DeepRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.render_on_reset = False

        self.state_dim = self.observation_space.shape
        self.action_dim = self.action_space.shape[0]

        self.name = 'DDPGEnv'
        #self.success_threshold = 2 #TODO

    def normalize_state(self, state):
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        #reward *= 1000  # often reward scaling
        return state, reward, done, info

    def reset(self):
        # here's a roundabout way to get it to plot on reset
        if self.render_on_reset:
            self.env.render('notebook')
        return self.env.reset()

def task_fn():
    env = PortfolioEnv(df=df_train, steps=steps, window_length=window, output_mode='EIIE')
    env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_test():
    env = PortfolioEnv(df=df_test, steps=420, window_length=window, output_mode='EIIE')
    env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_vali():
    env = PortfolioEnv(df=df_train, steps=1500, window_length=window, output_mode='EIIE')
    env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

# load
import pickle
import shutil

def save_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = '/Users/Morgans/Desktop/trading_system/video/%s-%s-model-%s.bin' % (
    agent_type, config.tag, agent.task.name)
    agent.save(save_file)
    print(save_file)


def load_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = '/Users/Morgans/Desktop/trading_system/video/%s-%s-model-%s.bin' % (
    agent_type, config.tag, agent.task.name)
    new_states = pickle.load(open(save_file, 'rb'))
    states = agent.worker_network.load_state_dict(new_states)


def load_stats_ddpg(agent):
    agent_type = agent.__class__.__name__
    online_stats_file = '/Users/Morgans/Desktop/trading_system/video/%s-%s-online-stats-%s.bin' % (
        agent_type, config.tag, agent.task.name)
    try:
        steps, rewards = pickle.load(open(online_stats_file, 'rb'))
    except FileNotFoundError:
        steps = []
        rewards = []
    df_online = pd.DataFrame(np.array([steps, rewards]).T, columns=['steps', 'rewards'])
    if len(df_online):
        df_online['step'] = df_online['steps'].cumsum()
        df_online.index.name = 'episodes'

    stats_file = '/Users/Morgans/Desktop/trading_system/video/%s-%s-all-stats-%s.bin' % (
    agent_type, config.tag, agent.task.name)
    try:
        stats = pickle.load(open(stats_file, 'rb'))
    except FileNotFoundError:
        stats = {}
    df = pd.DataFrame(stats["test_rewards"], columns=['rewards'])
    if len(df):
        #         df["steps"]=range(len(df))*50
        df.index.name = 'episodes'
    return df_online, df


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=torch.device('cpu'), dtype=torch.float32)
    return x

import logging
from agent import ProximalPolicyOptimization, DisjointActorCriticNet
from component import HighDimActionReplay, OrnsteinUhlenbeckProcess, AdaptiveParamNoiseSpec, hard_update, Replay
from utils.config import Config
from utils.tf_logger import Logger
import gym
import torch
from component.Experience import *
gym.logger.setLevel(logging.INFO)

class DDPGAgent:
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.worker_network = config.network_fn()
        self.target_network = config.network_fn()
        self.target_network.load_state_dict(self.worker_network.state_dict())
        self.actor_opt = config.actor_optimizer_fn(self.worker_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.worker_network.critic.parameters())
        self.replay = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.criterion = nn.MSELoss()
        self.total_steps = 0
        self.batch_size = 10
        self.state_normalizer = Normalizer(self.task.state_dim)
        self.reward_normalizer = Normalizer(1)
        self.memory = int(1e9)

    def perturb_actor_parameters(self, param_noise):
        """Apply parameter noise to actor model, for exploration"""
        hard_update(self.actor_perturbed, self.actor)
        params = self.actor_perturbed.state_dict()
        for name in params:
            if 'ln' in name:
                pass
            param = params[name]
            random = torch.randn(param.shape)
            if use_cuda:
                random = random.cuda()
            param += random * param_noise.current_stddev

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                                    param * self.config.target_network_mix)

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self.worker_network.state_dict(), f)

    #def store_transition(self, state, next_state, action, reward, done):
        #return self.replay.push(state, action, next_state, reward, done)

    def episode(self, deterministic=False, video_recorder=None):
        self.random_process.reset_states()
        state = self.task.reset()
        #state = self.state_normalizer.normalize(state)  * self.config.reward_scaling
        config = self.config
        actor = self.worker_network.actor
        #actor_perturbed = self.worker_network.actor
        critic = self.worker_network.critic
        target_actor = self.target_network.actor
        target_critic = self.target_network.critic
        steps = 0
        total_reward = 0.0
        while True:
            actor.eval()
            action = actor.predict(np.stack([state])).flatten()
            if not deterministic:
                action += self.random_process.sample()
            next_state, reward, done, info = self.task.step(action)
            #reward = self.reward_normalizer(reward)
            #next_state = self.state_normalizer.normalize(next_state)

            #obs = torch.from_numpy(state).unsqueeze(0)
            #inp = Variable(obs, requires_grad=False)  #.type(FloatTensor)
            #actor_perturbed.eval()

            #if param_noise is not None:
                #action = actor_perturbed.predict(np.stack([state])).flatten() #.data[0].cpu().numpy()

            #actor.train()

            #if self.random_process.sample() is not None:
                #action = action + self.random_process.sample()

            #action = Action(state, self.random_process.sample(), param_noise)
            #action = np.clip(action, 0, 1)
            #next_state, reward, done, info = self.task.step(action)

            if video_recorder is not None:
                video_recorder.capture_frame()
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))
            total_reward += reward

            #reward = self.reward_normalizer(reward)
            #next_state = self.state_normalizer.normalize(next_state) * self.config.reward_scaling

            # tensorboard logging
            prefix = 'test_' if deterministic else ''
            log_value(prefix + 'reward', reward, self.total_steps)
            #log_value(prefix + 'action', action, steps)
            #log_value('memory_size', self.replay.size(), self.total_steps)
            for key in info:
                log_value(key, info[key], self.total_steps)

            #reward = np.asscalar(self.reward_normalizer(np.array([reward])))

            if not deterministic:
                self.replay.add(state, action, reward, next_state, done)
                self.total_steps += 1

            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.replay.count() >= config.min_memory_size:
                #self.worker_network.train()
                #experiences = self.replay.sample()
                batch, w_id, eid = self.replay.getBatch(self.batch_size)
                states = np.zeros((self.batch_size, self.task.state_dim))
                rewards = np.zeros((self.batch_size,))
                action_mask = np.zeros((self.batch_size, self.task.action_dim))
                next_states = np.zeros((self.batch_size, self.task.state_dim))
                next_state_mask = np.zeros((self.batch_size,))

                for k, (s0, a, r, s1, done) in enumerate(batch):
                    states[k] = s0
                    rewards[k] = r
                    action_mask[k][a] = 1
                    # check terminal state
                    if not done:
                        next_states[k] = s1
                        next_state_mask[k] = 1

                states = tensor(states)
                actions = tensor(actions)
                rewards = tensor(rewards).unsqueeze(-1)
                terminals = tensor(terminals).unsqueeze(-1)
                next_states = tensor(next_states)
                q_next = target_critic.predict(next_states, target_actor.predict(next_states))
                #terminals = critic.to_torch_variable(dones).unsqueeze(1)
                #rewards = critic.to_torch_variable(rewards).unsqueeze(1)
                q_next = config.discount * q_next * (1 - terminals)
                q_next.add_(rewards)
                q_next = q_next.detach()
                q = critic.predict(states, actions)
                #critic_loss = self.criterion(q, q_next)  #TD error
                critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean() # critic_loss
                #  critic network updating
                critic.zero_grad()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                if self.config.gradient_clip:  #TODO
                    grad_critic = nn.utils.clip_grad_norm(self.worker_network.parameters(), self.config.gradient_clip)
                self.critic_opt.step()
                #  actor network updating
                actions = actor.predict(states, False)
                #var_actions = Variable(actions.data, requires_grad=True)
                q = critic.predict(states, actions)
                #q = critic.predict(states, var_actions)
                policy_loss = -q.mean()
                #policy_loss.backward()
                #q.backward(torch.ones(q.size()))
                actor.zero_grad()
                self.actor_opt.zero_grad()
                #actions.backward(-var_actions.grad.data)
                policy_loss.backward()
                if self.config.gradient_clip:
                    grad_actor = nn.utils.clip_grad_norm(self.worker_network.parameters(), self.config.gradient_clip)
                self.actor_opt.step()

                # tensorboard logging # TODO -q.sum(),critic_loss.cpu().data.numpy().squeeze()
                log_value('critic_loss', critic_loss.squeeze(), self.total_steps)
                log_value('policy_loss', policy_loss.sum(), self.total_steps)
                if config.gradient_clip:  # TODO
                    log_value('grad_critic', grad_critic, self.total_steps)
                    log_value('grad_actor', grad_actor, self.total_steps)
                self.soft_update(self.target_network, self.worker_network)

        return total_reward, steps

    def _step(self, state):
        actor = self.worker_network.actor
        #state = self.state_normalizer.normalize(state) * self.config.reward_scaling
        action = actor.predict(np.stack([state])).flatten()
        return action


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# task.state_dim, task.action_dim
from network.base_network import BasicNet


def to_torch_variable(x, dtype='float64'):
    if isinstance(x, Variable):
        return x
    if not isinstance(x, torch.FloatTensor):
        x = torch.from_numpy(np.asarray(x, dtype=dtype))
    return Variable(x)


class DeterministicActorNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_gate,
                 action_scale,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(DeterministicActorNet, self).__init__()

        stride_time = state_dim[1] - 1 - 2  #
        features = task.state_dim[0]
        h0 = 3
        h1 = 32
        self.conv1 = nn.Conv2d(features, h0, (3, 1))
        self.conv2 = nn.Conv2d(h0, h1, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d((h1 + 1), 1, (1, 1))

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h1+1)

        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)

    def forward(self, x):
        x = self.to_torch_variable(x)
        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]

        phi0 = self.non_linear(self.conv1(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv2(phi0))
        h = torch.cat([phi1, w0], 1)
        if self.batch_norm:
            h = self.bn2(h)
        action = self.conv3(h) # does not include cash account, add cash in next step.
        # add cash_bias before we softmax
        cash_bias_int = 1  #
        cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
        action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        if self.action_gate:
            action = self.action_scale * self.action_gate(action)
        #action = F.softmax(action)
        return action

    def predict(self, x, to_numpy=True):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y


class DeterministicCriticNet(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(DeterministicCriticNet, self).__init__()
        stride_time = state_dim[1] - 1 - 2  #
        self.features = features = task.state_dim[0]
        h0 = 2
        h1 = 20
        self.action = actions = action_dim - 1
        self.conv1 = nn.Conv2d(features, h0, (3, 1))
        self.conv2 = nn.Conv2d(h0, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer3 = nn.Linear((h1 + 2) * actions, 1)
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h1+2)
        self.batch_norm = batch_norm

        BasicNet.__init__(self, None, gpu, False)

    def forward(self, x, action):
        x = self.to_torch_variable(x)
        action = self.to_torch_variable(action)[:, None, None, :-1]  # remove cash bias

        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]

        phi0 = self.non_linear(self.conv1(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv2(phi0))
        h = torch.cat([phi1, w0, action], 1)
        if self.batch_norm:
            h = self.bn2(h)
        batch_size = x.size()[0]
        #action = self.non_linear(self.layer3(h))
        action = self.layer3(h.view((batch_size, -1)))
        #action = F.softmax(h.view((batch_size, -1)))
        return action

    def predict(self, x, action):
        return self.forward(x, action)


config = Config()
config.task_fn = task_fn
task = config.task_fn()
config.actor_network_fn = lambda: DeterministicActorNet(
    task.state_dim, task.action_dim, action_gate=None, action_scale=1.0, non_linear=F.relu,
    batch_norm=False, gpu=False)
config.critic_network_fn = lambda: DeterministicCriticNet(
    task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=False, gpu=False)
config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-6)
config.critic_optimizer_fn = \
    lambda params: torch.optim.Adam(params, lr=1e-5, weight_decay=0.001)
#config.replay_fn = lambda: HighDimActionReplay(memory_size=int(1e9), batch_size=64)
config.replay_fn = lambda: ReplayBuffer(buffer_size = int(1e9))
config.random_process_fn = \
    lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.15, sigma=0.25, sigma_min=0.001,
                                     n_steps_annealing=10000)
config.discount = 0.9
config.min_memory_size = 50
config.max_steps = 300000
config.max_episode_length = 3000
config.target_network_mix = 0.001
config.noise_decay_interval = 100000
config.gradient_clip = 10
config.min_epsilon = 0.1
config.reward_scaling = 100
config.test_interval = 20
config.test_repetitions = 1
config.save_interval = 200
config.episode_limit = 200
config.logger = Logger('/Users/Morgans/Desktop/trading_system/log', gym.logger)
config.tag = tag
agent = DDPGAgent(config)
agent




from utils.misc import run_episodes
agent.task._plot = agent.task._plot2 = None
try:
    run_episodes(agent)
except KeyboardInterrupt as e:
    save_ddpg(agent)
    raise (e)

plt.figure()
df_online, df = load_stats_ddpg(agent)
sns.regplot(x="step", y="rewards", data=df_online, order=1)
plt.show()
portfolio_return = (1+df_online.rewards.mean())
returns = task.unwrapped.src.data[0,:,:1]
ave_return = (1+returns).mean()
print(ave_return, portfolio_return)
agent.task.render('notebook')
agent.task.render('humman')
df_info = pd.DataFrame(agent.task.unwrapped.infos)
df_info[["portfolio_value", "market_value"]].plot(title = "price", fig=plt.gcf())
plt.show()

def test_algo(env, algo):
    #np.random.seed(0)
    algo.config.task = task_fn_test()
    state = env.reset()
    #state_normalizer = Normalizer(state.shape)
    #state = state_normalizer.normalize(state)*100
    done = False
    actions = []
    while not done:
        action = algo._step(state)
        state, reward, done, info = env.step(action)
        if done:
            break
        actions = getattr(action, 'value', action)
    df = pd.DataFrame(env.unwrapped.infos)
    #df.index = pd.to_datetime(df['date'] * 1e9)
    env.render(mode = 'notebook')
    env.render(mode = 'humman')
    return df['portfolio_value'], df, actions

#agent.task= task_fn_test()
#agent.episode(True)
portfolio_value, df_v, actions= test_algo(task_fn_vali(), agent)

portfolio_value, df_v, actions= test_algo(task_fn_test(), agent)
#df_v[["portfolio_value", "market_value"]].plot(title = "price", fig=plt.gcf())
df_v[["portfolio_value", "market_value"]]


