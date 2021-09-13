import os
# os.sys.path.append(os.path.abspath('.'))
# os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system/'))
from matplotlib import pyplot as plt
import sys
import matplotlib

# matplotlib.use('nbAgg', force=True)
matplotlib.rc('figure', figsize=[15, 10])
# matplotlib.use('TkAgg')
import seaborn as sns
import numpy as np
import threading
from numpy import random
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
# from utils.data import read_stock_history, index_to_date, date_to_index, normalize
import matplotlib
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from collections import Counter
import tempfile
import logging
import time
import seaborn as sns
from scipy.stats import norm

window = 10
root = os.getcwd()
steps = 128
import datetime

ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
save_path = root + '/log_TEST'
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
from wrappers import SoftmaxActions, TransposeHistory
from wrappers.concat import ConcatStates
from wrappers.logit import LogitActions

# df_train = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/ten_stock/poloniex_ten_sh.hf', key='train')
# df_test = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/ten_stock/poloniex_ten_sh.hf', key='test')
# path_data = root+'/HFT_data/financial_crisis/poloniex_fc.hf'
path_data = root + '/HFT_data/ETF/poloniex_fc.hf'
df_train = pd.read_hdf(path_data, key='train', encoding='utf-8')
df_test = pd.read_hdf(path_data, key='test', encoding='utf-8')
# df_train = pd.read_hdf('/tmp/pycharm_project_927/HFT_data/financial_crisis/poloniex_fc.hf', key='train')
# df_test = pd.read_hdf('/tmp/pycharm_project_927/HFT_data/financial_crisis/poloniex_fc.hf', key='test')


import gym


class DeepRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.render_on_reset = False

        self.state_dim = self.observation_space.shape
        self.action_dim = self.action_space.shape[0]

        self.name = 'DDPGEnv'
        # self.success_threshold = 2 #TODO

    def normalize_state(self, state):
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        # reward *= 1000  # often reward scaling
        return state, reward, done, info

    def reset(self):
        # here's a roundabout way to get it to plot on reset
        if self.render_on_reset:
            self.env.render('notebook')
        return self.env.reset()


def task_fn():
    env = PortfolioEnv(df=df_train, steps=128, window_length=window, output_mode='EIIE',
                       utility='Log', scale=True, scale_extra_cols=True, random_reset=True)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env


def task_fn_test():
    env = PortfolioEnv(df=df_test, steps=500, window_length=window, output_mode='EIIE',
                       utility='Log', scale=True, scale_extra_cols=True, random_reset=False)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env


def task_fn_vali():
    env = PortfolioEnv(df=df_train, steps=2000, window_length=window, output_mode='EIIE',
                       utility='Log', scale=True, scale_extra_cols=True, random_reset=False)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env


import pickle
import shutil


def save_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = root + '/video/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name)
    agent.save(save_file)
    return save_file


def load_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = root + '/video/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name)
    new_states = pickle.load(open(save_file, 'rb'))
    states = agent.worker_network.load_state_dict(new_states)
    return save_file


def load_stats_ddpg(agent):
    agent_type = agent.__class__.__name__
    online_stats_file = root + '/video/%s-%s-online-stats-%s.bin' % (
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
    stats_file = root + '/video/%s-%s-all-stats-%s.bin' % (
        agent_type, config.tag, agent.task.name)
    try:
        stats = pickle.load(open(stats_file, 'rb'))
    except FileNotFoundError:
        stats = {}
    df = pd.DataFrame(stats["test_rewards"], columns=['rewards'])
    if len(df):
        # df["steps"]=range(len(df))*50
        df.index.name = 'episodes'
    return df_online, df


import logging
from agent import ProximalPolicyOptimization, DisjointActorCriticNet
from component import HighDimActionReplay, OrnsteinUhlenbeckProcess, AdaptiveParamNoiseSpec, hard_update, \
    ddpg_distance_metric
from utils.config import Config
from utils.tf_logger import Logger
import gym
import torch
from utils.normalizer import Normalizer, StaticNormalizer

gym.logger.setLevel(logging.INFO)


def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=torch.device('cpu'), dtype=torch.float32)
    return x


import torch.multiprocessing as mp


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
        self.tag = 'general DDPG'

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(
                target_param * (1.0 - self.config.target_network_mix) + param * self.config.target_network_mix)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self.worker_network.state_dict(), f)

    def episode(self, deterministic=False, video_recorder=None):
        self.random_process.reset_states()
        state = self.task.reset()
        config = self.config
        actor = self.worker_network.actor
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
                # action = np.clip(action, 0, 1)
                # else:
                # action += max(self.epsilon, config.min_epsilon) * self.random_process.sample()
                # self.epsilon -= self.d_epsilon
                # action += self.random_process.sample()
                # actor.train()
                # action = action.data
                # action += torch.Tensor(self.random_process.sample())
            next_state, reward, done, info = self.task.step(action)
            # next_state = self.state_normalizer(next_state)

            if video_recorder is not None:
                video_recorder.capture_frame()
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))
            total_reward += reward
            # reward = self.reward_normalizer(reward)
            # next_state = self.state_normalizer.normalize(next_state) * self.config.reward_scaling

            # tensorboard logging
            prefix = 'test_' if deterministic else ''
            log_value(prefix + 'reward', reward, self.total_steps)
            # log_value(prefix + 'action', action, steps)
            # log_value('memory_size', self.replay.size(), self.total_steps)
            for key in info:
                log_value(key, info[key], self.total_steps)
            # reward = self.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1

            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.replay.size() >= config.min_memory_size:
                self.worker_network.train()
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = tensor(states)
                actions = tensor(actions)
                rewards = tensor(rewards).unsqueeze(-1)
                mask = tensor(1 - terminals).unsqueeze(-1)
                next_states = tensor(next_states)
                q_next = target_critic.predict(next_states, target_actor.predict(next_states))
                q_next = config.discount * q_next * mask
                q_next.add_(rewards)
                q_next = q_next.detach()
                q = critic.predict(states, actions)
                critic_loss = self.criterion(q, q_next)
                # TD error
                # critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean() # critic_loss/ TD_error
                #  critic network updating
                critic.zero_grad()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                grad_critic = nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
                self.critic_opt.step()

                #  actor network updating
                Actions = actor.predict(states, False)
                # var_actions = Variable(actions.data, requires_grad=True)
                q = critic.predict(states, Actions)
                # q = critic.predict(states, var_actions)
                policy_loss = -q.mean()
                # q.backward(torch.ones(q.size()))
                actor.zero_grad()
                self.actor_opt.zero_grad()
                policy_loss.backward()
                grad_actor = nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip)
                self.actor_opt.step()

                # tensorboard logging # TODO -q.sum(),critic_loss.cpu().data.numpy().squeeze()
                log_value('critic_loss', critic_loss.sum(), self.total_steps)
                log_value('policy_loss', policy_loss.sum(), self.total_steps)
                if config.gradient_clip:
                    log_value('grad_critic', grad_critic, self.total_steps)
                    log_value('grad_actor', grad_actor, self.total_steps)
                self.soft_update(self.target_network, self.worker_network)

        return total_reward, steps

    def _step(self, state, to_numpy=True):
        actor = self.worker_network.actor
        # state = self.state_normalizer.normalize(state) * self.config.reward_scaling
        action = actor.predict(state, to_numpy)
        return action


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.base_network import BasicNet


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
        h0 = 8
        h2 = 16
        h1 = 32
        # self.conv0 = nn.Conv2d(features, h0, (3, 3), stride = (1,1), padding=(1,1)) # input 64*5 *50 *10 out 64* 48 *8
        self.conv1 = nn.Conv2d(features, h2, (3, 1))  # input 64 * 50 * 10   output 64 *48 *8
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d((h1 + 1), 1, (1, 1))
        # self.out = nn.Linear(5, 5)

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1 + 1)

        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)

    def to_torch_variable(self, x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        return Variable(x)

    def forward(self, x):
        x = self.to_torch_variable(x)
        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]

        # phi0 = self.non_linear(self.conv0(x))
        # if self.batch_norm:
        #     phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(x))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        h = torch.cat([phi2, w0], 1)
        if self.batch_norm:
            h = self.bn3(h)
        action = self.conv3(h)  # does not include cash account, add cash in next step.
        # add cash_bias before we softmax
        cash_bias_int = 0  #
        cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
        action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        if self.action_gate:
            action = self.action_scale * self.action_gate(action)
        # action = self.non_linear(self.out(action))
        # action /= action.sum()
        # action = F.softmax(self.out(action), dim =1)
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
        h0 = 8
        h2 = 16
        h1 = 32
        self.action = actions = action_dim - 1
        # self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1,1))
        self.conv1 = nn.Conv2d(features, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer3 = nn.Linear((h1 + 2) * actions, 1)
        self.non_linear = non_linear
        # self.fc = nn.Dropout(0.3)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1 + 2)
        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)

    def to_torch_variable(self, x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        return Variable(x)

    def forward(self, x, action):
        x = self.to_torch_variable(x)
        action = self.to_torch_variable(action)[:, None, None, :-1]  # remove cash bias

        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]

        # phi0 = self.non_linear(self.conv0(x))
        # if self.batch_norm:
        #     phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(x))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        h = torch.cat([phi2, w0, action], 1)
        if self.batch_norm:
            h = self.bn3(h)
        batch_size = x.size()[0]
        # action = self.non_linear(self.layer3(h))
        action = self.layer3(h.view((batch_size, -1)))
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
config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-5)
config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4, weight_decay=0.001)
# config.replay_fn = lambda: ReplayMemory(capacity=int(1e9))
config.replay_fn = lambda: HighDimActionReplay(memory_size=3000, batch_size=64)
config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.3, sigma=0.3, sigma_min=0.01,
                                                            n_steps_annealing=10000)

config.discount = 0.95
config.min_memory_size = 1000
config.max_steps = 100000
config.max_episode_length = 10000
config.target_network_mix = 0.001
config.noise_decay_interval = 10000
config.gradient_clip = 20
config.min_epsilon = 0.1
config.reward_scaling = 1
config.test_interval = 50
config.test_repetitions = 1
config.save_interval = config.episode_limit = 200
# config.logger = Logger('/Users/Morgans/Desktop/trading_system/log', gym.logger)
config.logger = Logger(root + '/log', gym.logger)
config.tag = tag
agent = DDPGAgent(config)
# agent
# torch.save(agent.worker_network.state_dict(), save_file)
# agent2.worker_network.load_state_dict(torch.load('agent1'))

# from utils.misc import run_episodes
# agent.task._plot = agent.task._plot2 = None
# try:
#    run_episodes(agent)
# except KeyboardInterrupt as e:
#    save_ddpg(agent)
#    raise (e)
# log_dir = '/Users/Morgans/Desktop/trading_system/video/DDPGAgent-ddpg-agent.pth'
log_dir = '/Users/Morgans/Desktop/trading_system/video/ETF weights/DDPGAgent-ddpg-cnn-agent-ETF-win10-etf_chn2.pth'
agent.worker_network.load_state_dict(torch.load(log_dir))


def test_algo(env, algo):
    # algo.config.task = task_fn_test()
    state = env.reset()
    done = False
    actions = []
    while not done:
        action = algo._step(np.stack([state])).flatten()
        actions.append(action)
        # state, reward, done, info = env.step(action)
        state, _, done, _ = env.step(action)
        if done:
            break
        # actions = getattr(action, 'value', action)
    df = pd.DataFrame(env.unwrapped.infos)
    # df.index = pd.to_datetime(df['date'] * 1e9)
    env.render(mode='notebook')
    env.render(mode='humman')
    return df['portfolio_value'], df, actions


portfolio_value, df_v, actions = test_algo(task_fn_test(), agent)

# df_v[["portfolio_value", "market_value"]].plot(title = "price", fig=plt.gcf())

# df_v[["portfolio_value", "market_value"]]
# df_v['CVaR'].plot()
# plt.show()


from utils.configration import Configration
from Environment.ENV import PPortfolioEnv
from utils.util import MDD, sharpe, softmax
from wrappers import RobSoftmaxActions, RobTransposeHistory
from wrappers.concat import RobConcatStates


class DDeepRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.render_on_reset = False

        self.state_dim = self.observation_space.shape
        self.action_dim = self.action_space.shape[0]

        self.name = 'ENV'
        # self.success_threshold = 2 #TODO

    def normalize_state(self, state):
        return state

    def step(self, action1, action2):
        state, reward_gt, reward_at, done, info, z = self.env.step(action1, action2)
        # reward *= 1000  # often reward scaling
        return state, reward_gt, reward_at, done, info, z

    def reset(self):
        # here's a roundabout way to get it to plot on reset
        if self.render_on_reset:
            self.env.render('notebook')
        return self.env.reset()


def task_fn_H():
    env = PPortfolioEnv(df=df_train, steps=128, window_length=window, output_mode='EIIE',
                        gamma=2, c=0.08, trading_cost=0.0025, utility='Log', scale=True,
                        scale_extra_cols=True, random_reset=True)
    env = RobTransposeHistory(env)
    env = RobConcatStates(env)
    env = RobSoftmaxActions(env)
    env = DDeepRLWrapper(env)
    return env


def task_fn_test_H():
    env = PPortfolioEnv(df=df_test, steps=500, window_length=window, output_mode='EIIE',
                        gamma=2, c=0.13, trading_cost=0.0025, utility='Log', scale=True,
                        scale_extra_cols=True, random_reset=False)
    env = RobTransposeHistory(env)
    env = RobConcatStates(env)
    env = RobSoftmaxActions(env)
    env = DDeepRLWrapper(env)
    return env


def task_fn_vali_H():
    env = PPortfolioEnv(df=df_train, steps=2000, window_length=window, output_mode='EIIE',
                        gamma=2, c=0.13, trading_cost=0.0025, utility='Log', scale=True,
                        scale_extra_cols=True, random_reset=False)
    env = RobTransposeHistory(env)
    env = RobConcatStates(env)
    env = RobSoftmaxActions(env)
    env = DDeepRLWrapper(env)
    return env


class HiAgent(mp.Process):
    def __init__(self, config, agent):
        self.config = config
        self.agent = agent
        # self.worker_network = config.network_fn()
        self.task = task_fn_H()
        self.worker_network_H = config.network_fn_H()
        self.target_network_H = config.network_fn_H()  # adding target network
        # self.target_network.load_state_dict(self.worker_network.state_dict())
        self.target_network_H.load_state_dict(self.worker_network_H.state_dict())
        self.random_process = config.random_process_fn()
        self.criterion = nn.MSELoss()
        self.total_steps = 0
        self.alpha = 0.05
        self.replay_Hierachical = config.replay_Hierachical()
        self.actor_high_opt = config.actor_optimizer(self.worker_network_H.actor.parameters())
        self.critic_high_opt = config.critic_optimizer(self.worker_network_H.critic.parameters())
        self.tag = 'DDPG-Hi'
        super().__init__()

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self.worker_network_H.state_dict(), f)

    def episode(self, deterministic=False, video_recorder=None):
        self.random_process.reset_states()
        state = self.task.reset()
        # state = self.state_normalizer(state)
        # state = self.min_max_scale(state)
        config = self.config
        actor_high = self.worker_network_H.actor
        critic_high = self.worker_network_H.critic
        target_actor_high = self.target_network_H.actor
        target_critic_high = self.target_network_H.critic
        steps = 0
        total_reward_gt = 0.0
        total_reward_at = 0.0
        while True:
            # actor.eval()
            actor_high.eval()
            gt = self.agent._step(np.stack([state])).flatten()  # TODO check agent output
            at = actor_high.predict(np.stack([state]), np.stack([gt])).flatten()
            if not deterministic:
                at += self.random_process.sample()
            next_state, reward_gt, reward_at, done, info, z = self.task.step(gt, at)
            # next_state = self.min_max_scale(next_state)
            if video_recorder is not None:
                video_recorder.capture_frame()
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))

            total_reward_gt += reward_gt
            total_reward_at += reward_at
            self.total_steps += 1
            # tensorboard logging
            prefix = 'test_' if deterministic else ''
            log_value(prefix + 'reward_gt', reward_gt, self.total_steps)
            log_value(prefix + 'reward_at', reward_at, self.total_steps)
            for key in info:
                log_value(key, info[key], self.total_steps)
            # if z == 1:
            self.replay_Hierachical.feed([state, gt, at, reward_at, next_state, int(done)])
            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.replay_Hierachical.size() >= config.min_memory_size:
                self.worker_network_H.train()
                # actor.eval()
                experiences2 = self.replay_Hierachical.sample()
                states_H, actions_gt, actions_at, rewards_at, next_states_H, terminals_H = experiences2
                states_H = tensor(states_H)
                actions_gt = tensor(actions_gt)
                actions_at = tensor(actions_at)
                rewards_at = tensor(rewards_at).unsqueeze(-1)
                mask_H = tensor(1 - terminals_H).unsqueeze(-1)
                next_states_H = tensor(next_states_H)
                gt_next = self.agent._step(next_states_H, False)
                gt_next = gt_next.detach()
                # gt_next = gt_next.detach()
                # gt_next = Variable(gt_next.data, requires_grad=False)
                q_next_H = target_critic_high.predict(next_states_H, gt_next,
                                                      target_actor_high.predict(next_states_H, gt_next))
                q_next_H = config.discount * q_next_H * mask_H
                q_next_H.add_(rewards_at)
                q_next_H = q_next_H.detach()
                q_H = critic_high.predict(states_H, actions_gt, actions_at)
                critic_loss_H = self.criterion(q_H, q_next_H)  # TODO adding new criterion for Hierachical #TD error
                #  critic_hight network updating
                critic_high.zero_grad()
                self.critic_high_opt.zero_grad()
                critic_loss_H.backward()
                grad_critic_H = nn.utils.clip_grad_norm_(critic_high.parameters(), config.gradient_clip)
                self.critic_high_opt.step()

                #  actor_hight network updating
                Actions_H = actor_high.predict(states_H, actions_gt, False)
                # var_actions_H = Variable(Actions_H.data, requires_grad=True)
                qq_H = critic_high.predict(states_H, actions_gt, Actions_H)
                policy_loss_H = -qq_H.mean()
                actor_high.zero_grad()
                self.actor_high_opt.zero_grad()
                policy_loss_H.backward()
                grad_actor_H = nn.utils.clip_grad_norm_(actor_high.parameters(), config.gradient_clip)
                self.actor_high_opt.step()

                log_value('critic_loss_H', critic_loss_H.sum(), self.total_steps)
                log_value('policy_loss_H', policy_loss_H.sum(), self.total_steps)
                if config.gradient_clip:
                    log_value('grad_critic_H', grad_critic_H, self.total_steps)
                    log_value('grad_actor_H', grad_actor_H, self.total_steps)
                self.soft_update(self.target_network_H, self.worker_network_H)

        return total_reward_gt, total_reward_at, steps

    def _step(self, state):
        gt = self.agent._step(np.stack([state])).flatten()
        at = self.worker_network_H.actor.predict(np.stack([state]), np.stack([gt])).flatten()
        return gt, at


class Hierachicalagent(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_gate,
                 action_scale,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(Hierachicalagent, self).__init__()

        stride_time = state_dim[1] - 1 - 2  #
        features = task.state_dim[0]
        h0 = 8
        h2 = 16
        h1 = 32
        # self.conv0 = nn.Conv2d(features, h0, (3, 3), stride = (1,1), padding=(1,1)) # input 64*5 *50 *10 out 64* 48 *8
        self.conv1 = nn.Conv2d(features, h1, (3, 1))  # input 64 * 50 * 10   output 64 *48 *8
        self.conv2 = nn.Conv2d(h1, h2, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d((h2 + 1), 1, (1, 1))
        # self.out = nn.Linear(5, 5)

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h1)
            self.bn3 = nn.BatchNorm2d(h2 + 1)
            self.bn4 = nn.BatchNorm2d(h2)

        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)

    def to_torch_variable(self, x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        return Variable(x)

    def forward(self, x, w1):
        x = self.to_torch_variable(x)
        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]  # remove a_t
        w1 = self.to_torch_variable(w1)[:, None, None, :-1]
        # phi0 = self.non_linear(self.conv0(x))
        # if self.batch_norm:
        #     phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(x))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        h = torch.cat([phi2, w1], 1)
        if self.batch_norm:
            h = self.bn3(h)
        action = self.conv3(h)  # does not include cash account, add cash in next step.
        # add cash_bias before we softmax
        cash_bias_int = 0
        cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
        action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        if self.action_gate:
            action = self.action_scale * self.action_gate(action)
        # action = self.non_linear(self.out(action))
        # action = F.softmax(action, dim=1)
        return action

    def predict(self, x, w1, to_numpy=True):
        y = self.forward(x, w1)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y


class Critichierachical(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(Critichierachical, self).__init__()
        stride_time = state_dim[1] - 1 - 2  #
        self.features = features = task.state_dim[0]
        h0 = 8
        h2 = 16
        h1 = 32
        self.action = actions = action_dim - 1
        # self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1,1))
        self.conv1 = nn.Conv2d(features, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer3 = nn.Linear((h1 + 2) * actions, 1)  # TODO adding action2
        self.non_linear = non_linear
        self.fc = nn.Dropout(0.0)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1 + 2)
            self.bn4 = nn.BatchNorm2d(h1)
        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)

    def to_torch_variable(self, x, dtype='float32'):
        if isinstance(x, Variable):
            return x
        if not isinstance(x, torch.FloatTensor):
            x = torch.from_numpy(np.asarray(x, dtype=dtype))
        return Variable(x)

    def forward(self, x, action1, action2):
        x = self.to_torch_variable(x)
        action1 = self.to_torch_variable(action1)[:, None, None, :-1]  # remove cash bias
        action2 = self.to_torch_variable(action2)[:, None, None, :-1]
        w0 = x[:, :1, :1, :]
        # w1 = x[:, :1, 1:2, :] # weights from last step
        x = x[:, :, 1:, :]

        # phi0 = self.non_linear(self.conv0(x))
        # if self.batch_norm:
        #     phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(x))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        h = torch.cat([phi2, action1, action2], 1)
        if self.batch_norm:
            h = self.bn3(h)
        batch_size = x.size()[0]
        # action = self.non_linear(self.layer3(h))
        action = self.layer3(h.view((batch_size, -1)))
        return action

    def predict(self, x, action1, action2):
        return self.forward(x, action1, action2)


from component import HighDimActionReplay, OrnsteinUhlenbeckProcess, AdaptiveParamNoiseSpec, hard_update, \
    ddpg_distance_metric, RobHighDimActionReplay

config1 = Configration()
config1.task_fn_H = task_fn_H
task_H = config1.task_fn_H()
config1.actor_high = lambda: Hierachicalagent(
    task_H.state_dim, task_H.action_dim, action_gate=None, action_scale=1.0, non_linear=F.relu,
    batch_norm=False, gpu=False)
config1.critic_high = lambda: Critichierachical(
    task_H.state_dim, task_H.action_dim, non_linear=F.relu, batch_norm=False, gpu=False)
# config.actor_network_fn = lambda: DeterministicActorNet(
#    task.state_dim, task.action_dim, action_gate=None, action_scale=1.0, non_linear=F.tanh,
#    batch_norm=True, gpu=False)
# config.critic_network_fn = lambda: DeterministicCriticNet(
#    task.state_dim, task.action_dim, non_linear=F.tanh, batch_norm=True, gpu=False)
# config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
config1.network_fn_H = lambda: DisjointActorCriticNet(config1.actor_high, config1.critic_high)
# config1.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-6, weight_decay=1e-4)
# config1.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-5, weight_decay=1e-4)
config1.actor_optimizer \
    = lambda params: torch.optim.Adam(params, lr=1e-5)
config1.critic_optimizer \
    = lambda params: torch.optim.Adam(params, lr=1e-4, weight_decay=0.001)

config1.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=task_H.action_dim, theta=0.3, sigma=0.3,
                                                             sigma_min=0.01, n_steps_annealing=10000)
config1.replay_Hierachical = lambda: RobHighDimActionReplay(memory_size=10000, batch_size=64)
config1.discount = 0.95
config1.min_memory_size = 1000
config1.max_steps = 100000
config1.max_episode_length = 3000
config1.target_network_mix = 0.001
config1.noise_decay_interval = 10000
config1.gradient_clip = 10
config1.min_epsilon = 0.1
config1.reward_scaling = 1
config1.test_interval = 50
config1.test_repetitions = 1
config1.save_interval = config1.episode_limit = 150
# config1.logger = Logger('/Users/Morgans/Desktop/trading_system/log', gym.logger)
config1.logger = Logger(root + '/log', gym.logger)
config1.tag = 'Hi-Agent'
agent2 = HiAgent(config1, agent)


def training(agent):
    config = agent.config
    window_size = window
    ep = 0
    # actions = []
    rewards_gt = []
    rewards_at = []
    steps = []
    avg_test_rewards_gt = []
    avg_test_rewards_at = []
    agent_type = agent.__class__.__name__
    while True:
        ep += 1
        reward_gt, reward_at, step = agent.episode()
        rewards_gt.append(reward_gt)
        rewards_at.append(reward_at)
        steps.append(step)
        # avg_reward_gt = np.mean(rewards_gt[-window_size:])
        # avg_reward_at = np.mean(rewards_at[-window_size:])
        avg_reward_gt = np.mean(rewards_gt)
        avg_reward_at = np.mean(rewards_at)
        config.logger.info(
            'episode %d, reward_gt %f, reward_at %f,   avg reward_gt %f, avg reward_at %f, total steps %d, episode step %d' % (
                ep, reward_gt, reward_at, avg_reward_gt, avg_reward_at, agent.total_steps, step))

        if config.save_interval and ep % config.save_interval == 0:
            with open(root + '/video/%s-%s-online-stats-%s.bin' % (
                    agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump([steps, rewards_gt, rewards_at], f)

        if config.render_episode_freq and ep % config.render_episode_freq == 0:
            video_recoder = gym.wrappers.monitoring.video_recorder.VideoRecorder(
                env=agent.task.env,
                base_path=root + '/video/%s-%s-%s-%d' % (agent_type, config.tag, agent.task.name, ep))
            agent.episode(True, video_recoder)
            video_recoder.close()

        if config.episode_limit and ep > config.episode_limit:
            break

        if config.max_steps and agent.total_steps > config.max_steps:
            break

        if config.test_interval and ep % config.test_interval == 0:
            config.logger.info('Testing...')
            agent.save(root + '/video/%s-%s-model-%s.bin' % (agent_type, config.tag, agent.task.name))
            test_rewards_gt = []
            test_rewards_at = []
            for _ in range(config.test_repetitions):
                test_rewards_gt.append(agent.episode(True)[0])
                test_rewards_at.append(agent.episode(True)[1])
            avg_reward_gt = np.mean(test_rewards_gt)
            avg_reward_at = np.mean(test_rewards_at)
            avg_test_rewards_gt.append(avg_reward_gt)
            avg_test_rewards_at.append(avg_reward_at)
            config.logger.info('Avg reward_gt %f, Avg reward_at %f' % (avg_reward_gt, avg_reward_at))
            with open(root + '/video/%s-%s-all-stats-%s.bin' % (agent_type, config.tag, agent.task.name), 'wb') as f:
                pickle.dump({'rewards_gt': rewards_gt,
                             'rewards_at': rewards_at,
                             'steps': steps,
                             'test_rewards_gt': avg_test_rewards_gt,
                             'test_rewards_at': avg_test_rewards_at}, f)
            # if avg_reward > config.success_threshold:
            # break

    return steps, rewards_gt, rewards_at, avg_test_rewards_gt, avg_test_rewards_at


agent2.task._plot = agent2.task._plot2 = agent2.task._plot3 = agent2.task._plot4 = agent2.task._plot5 = None

# log_dir_H = '/Users/Morgans/Desktop/trading_system/video/ETF weights/HiAgent-ddpg_cvar_win10_etf.pth'
# agent2.worker_network_H.load_state_dict(torch.load(log_dir_H))

try:
    training(agent2)
except KeyboardInterrupt as e:
    save_ddpg(agent2)
    raise (e)


def test_performance(env, agent):
    # algo.config.task = task_fn_test()
    state = env.reset()
    done = False
    actions = []
    while not done:
        action1, action2 = agent._step(state)
        state, reward_gt, reward_at, done, info, z = env.step(action1, action2)
        if z == 0:
            actions.append(action1)
        else:
            actions.append(action2)
        if done:
            break
        # actions = getattr(action, 'value', action)
    df = pd.DataFrame(env.unwrapped.infos)
    df.index = pd.to_datetime(df['date'] * 1e9)
    env.render(mode='notebook')
    env.render(mode='humman')  # TODO check the portfolio value
    return df['portfolio_value'], df, actions

portfolio_value, df_comb, actions = test_performance(task_fn_H(), agent2)
portfolio_value, df_comb, actions = test_performance(task_fn_vali_H(), agent2)
portfolio_value, df_comb, actions = test_performance(task_fn_test_H(), agent2)

# df_v[["portfolio_value", "market_value"]].plot(title = "price", fig=plt.gcf())

df_comb[["portfolio_value", "market_value"]].plot()
plt.show()
df_comb.plot(y=['CVaR_DDPG', 'CVaR'], use_index=True)
plt.show()
df_comb.plot(y=['Sharp ratio', 'Sharp ratio DDPG'], use_index=True)
plt.show()

# log_dir_H = '/Users/Morgans/Desktop/trading_system/video/HiAgent-ddpg_cvar_win20_etf.pth'
# log_dir_H = root+ '/video/HiAgent-ddpg_sharp.pth'
# agent2.save(log_dir_H)
