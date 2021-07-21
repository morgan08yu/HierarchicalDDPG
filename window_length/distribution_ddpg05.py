import os
#os.sys.path.append(os.path.abspath('.'))
#os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system/'))
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
window = 5
# os.chdir('../')
root = os.getcwd()
steps = 128
import datetime
ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
save_path = root+'/log_TEST'
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

from Environment.AlphaEnv import PortfolioEnv
from utils.util import MDD, sharpe, softmax
from wrappers import SoftmaxActions, TransposeHistory # ConcatStates
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
    env = PortfolioEnv(df=df_train, steps=128, window_length=window, utility='Log', output_mode='EIIE', gamma=2,
                       scale=True, scale_extra_cols=True, random_reset=True)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    # env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_test():
    env = PortfolioEnv(df=df_test, steps=500, window_length=window, utility='Log', output_mode='EIIE', gamma=2,
                       scale=True, scale_extra_cols=True, random_reset=False)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    # env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_vali():
    env = PortfolioEnv(df=df_train, steps=2000, window_length=window, utility='Log', output_mode='EIIE', gamma=2,
                       scale=True, scale_extra_cols=True, random_reset=False)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env


import pickle
import shutil

def save_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = root+'/video/%s-%s-model-%s.bin' % (
    agent_type, config.tag, agent.task.name)
    agent.save(save_file)
    print(save_file)


def load_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = root +'/video/%s-%s-model-%s.bin' % (
    agent_type, config.tag, agent.task.name)
    new_states = pickle.load(open(save_file, 'rb'))
    states = agent.worker_network.load_state_dict(new_states)


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
from component import HighDimActionReplay, OrnsteinUhlenbeckProcess, AdaptiveParamNoiseSpec, hard_update, ddpg_distance_metric
from utils.config import Config
from utils.tf_logger import Logger
import gym
import torch
from scipy.stats import norm
from utils.normalizer import Normalizer, StaticNormalizer
gym.logger.setLevel(logging.INFO)

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    x = np.asarray(x, dtype=np.float)
    x = torch.tensor(x, device=torch.device('cpu'), dtype=torch.float32)
    return x

import torch.multiprocessing as mp

class DDPGAgent(mp.Process):
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
        self.actor = self.worker_network.actor
        self.critic = self.worker_network.critic
        self.target_actor = self.target_network.actor
        self.target_critic = self.target_network.critic
        self.state_normalizer = StaticNormalizer(self.task.state_dim)
        self.reward_normalizer = StaticNormalizer(1)
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.3, adaptation_coefficient = 1.1)
        self.alpha = 0.9
        self.error = 1e-8

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix)+param * self.config.target_network_mix)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def save(self, file_name):
        with open(file_name, 'wb') as f:
            torch.save(self.worker_network.state_dict(), f)

    def episode(self, deterministic=False, video_recorder=None):
        self.random_process.reset_states()
        state = self.task.reset()
        #state = self.state_normalizer(state)
        config = self.config
        steps = 0
        total_reward = 0.0
        while True:
            self.actor.eval()
            action = self.actor.predict(np.stack([state])).flatten()
            if not deterministic:
                action += self.random_process.sample()
                #action = np.clip(action, 0, 1)
                #else:
                    #action += max(self.epsilon, config.min_epsilon) * self.random_process.sample()
                    #self.epsilon -= self.d_epsilon
                #action += self.random_process.sample()
                #actor.train()
                #action = action.data
                #action += torch.Tensor(self.random_process.sample())
            next_state, reward, done, info = self.task.step(action)

            if video_recorder is not None:
                video_recorder.capture_frame()
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))
            total_reward += reward

            # tensorboard logging
            prefix = 'test_' if deterministic else ''
            log_value(prefix + 'reward', reward, self.total_steps)
            # log_value(prefix + 'action', action, steps)
            # log_value('memory_size', self.replay.size(), self.total_steps)
            for key in info:
                log_value(key, info[key], self.total_steps)

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
                mask = tensor(1-terminals).unsqueeze(-1)
                next_states = tensor(next_states)
                q_next_raw = self.target_critic.predict(next_states, self.target_actor.predict(next_states))
                mu = q_next_raw[:, 0].unsqueeze(-1)
                sigma = q_next_raw[:, 1].unsqueeze(-1)
                mu_t = config.discount * mu * mask + self.error
                mu_t.add_(rewards)
                sigma_t = config.discount**2 * sigma * mask + self.error
                mu_t.detach()
                sigma_t.detach()

                q = self.critic.predict(states, actions)
                mu_p = q[:, 0].unsqueeze(-1) + self.error
                sigma_p = q[:, 1].unsqueeze(-1) + self.error
                critic_loss = torch.log(torch.sqrt(sigma_p/sigma_t))+(sigma_t + (mu_t-mu_p).pow(2))/(sigma_p.mul(2))
                # critic_loss = torch.pow(torch.abs(mu_t-mu_p), 2) + (sigma_t+sigma_p-2*torch.sqrt(sigma_p*sigma_t))

                cl = critic_loss.mean()
                # critic_loss = self.criterion(q[:, 0].unsqueeze(-1), q_next)
                # TD error
                # critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean() # critic_loss/ TD_error
                #  critic network updating
                self.critic.zero_grad()
                self.critic_opt.zero_grad()
                # critic_loss.backward()
                cl.backward()
                grad_critic = nn.utils.clip_grad_norm_(self.critic.parameters(), config.gradient_clip)
                self.critic_opt.step()

                #  actor network updating
                Actions = self.actor.predict(states, False)
                # var_actions = Variable(actions.data, requires_grad=True)
                score = self.critic.predict(states, Actions)
                Amu = score[:, 0].unsqueeze(-1) + self.error
                Asigma = score[:, 1].unsqueeze(-1) + self.error
                # q = critic.predict(states, var_actions)
                qq = Amu - self.alpha**-1 * norm.pdf(norm.ppf(self.alpha))*torch.sqrt(Asigma)
                # qq = Amu - norm.pdf(self.alpha)/norm.cdf(self.alpha) *torch.sqrt(Asigma)

                policy_loss = -qq.mean()
                # q.backward(torch.ones(q.size()))
                self.actor.zero_grad()
                self.actor_opt.zero_grad()
                policy_loss.backward()
                grad_actor = nn.utils.clip_grad_norm_(self.actor.parameters(), config.gradient_clip)
                self.actor_opt.step()

                # tensorboard logging # TODO -q.sum(),critic_loss.cpu().data.numpy().squeeze()
                log_value('critic_loss', critic_loss.sum(), self.total_steps)
                log_value('policy_loss', policy_loss.sum(), self.total_steps)
                if config.gradient_clip:
                    log_value('grad_critic', grad_critic, self.total_steps)
                    log_value('grad_actor', grad_actor, self.total_steps)
                self.soft_update(self.target_actor, self.actor)
                self.soft_update(self.target_critic, self.critic)

        return total_reward, steps

    def _step(self, state):
        action = self.actor.predict(np.stack([state])).flatten()
        return action


import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.base_network import BasicNet


class DeterministicActorNetCVaR(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 action_gate,
                 action_scale,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(DeterministicActorNetCVaR, self).__init__()

        stride_time = state_dim[1] - 1 - 2  #
        features = task.state_dim[0]
        h0 = 8
        h2 = 16
        h1 = 32
        # self.conv0 = nn.Conv2d(features, h0, (3, 3), stride = (1,1), padding=(1,1)) # input 64*5 *50 *10 out 64* 48 *8
        self.conv1 = nn.Conv2d(features, h2, (3, 1)) # input 64 * 50 * 10   output 64 *48 *8
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d((h1 + 1), 1, (1, 1))

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1+1)

        self.batch_norm = batch_norm
        #BasicNet.__init__(self, None, gpu, False)

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
        #m = nn.Dropout(p=0.25)
        #phi2 = m(phi2)
        h = torch.cat([phi2, w0], 1)
        if self.batch_norm:
            h = self.bn3(h)
        action = self.conv3(h) # does not include cash account, add cash in next step.
        # add cash_bias before we softmax
        cash_bias_int = 0  #
        cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
        action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        if self.action_gate:
            action = self.action_scale * self.action_gate(action)
        # action = self.out(action)
        action = F.softmax(action, dim=1)
        return action

    def predict(self, x, to_numpy=True):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y


class DeterministicCriticNetCVaR(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(DeterministicCriticNetCVaR, self).__init__()
        stride_time = state_dim[1] - 1 - 2  #
        self.features = features = task.state_dim[0]
        h0 = 8
        h2 = 16
        h1 = 32
        self.action = actions = action_dim - 1
        # self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1,1))
        self.conv1 = nn.Conv2d(features, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer3 = nn.Linear((h1 + 2) * actions, 1) # mu
        self.layer4 = nn.Linear((h1 + 2) * actions, 1) # variance
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1+2)
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
        action = self.to_torch_variable(action)[:, None, None, 1:]  # remove cash bias

        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]

        # phi0 = self.non_linear(self.conv0(x))
        # if self.batch_norm:
        #     phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(x))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        # m = nn.Dropout(p=0.25)
        # phi2 = m(phi2)
        h = torch.cat([phi2, w0, action], 1)
        if self.batch_norm:
            h = self.bn3(h)
        batch_size = x.size()[0]
        # action = self.non_linear(self.layer3(h))
        mu = self.layer3(h.view((batch_size, -1)))
        # h2 = self.non_linear(mu)
        var = F.softplus(self.layer4(h.view((batch_size, -1)))) # output is 2 dimensional---variance
        # mu = self.layer4(h2)
        actt = torch.cat([mu, var], 1)
        # ac = F.softplus(act)
        return actt

    def predict(self, x, action):
        return self.forward(x, action)


config = Config()
config.task_fn = task_fn
task = config.task_fn()
config.actor_network_fn = lambda: DeterministicActorNetCVaR(
    task.state_dim, task.action_dim, action_gate=None, action_scale=1.0, non_linear=F.relu,
    batch_norm=False, gpu=False)
config.critic_network_fn = lambda: DeterministicCriticNetCVaR(
    task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=False, gpu=False)
config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-5)
config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4, weight_decay=0.01)
# config.replay_fn = lambda: ReplayMemory(capacity=int(1e9))
config.replay_fn = lambda: HighDimActionReplay(memory_size=int(1e6), batch_size=32)
config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size = task.action_dim, theta=0.3,
                                                            sigma=0.3, sigma_min=0.01, n_steps_annealing=10000)

config.discount = 0.99
config.min_memory_size = 1000
config.max_steps = 100000
config.max_episode_length = 3000
config.target_network_mix = 0.001
config.noise_decay_interval = 10000
config.gradient_clip = 20
config.min_epsilon = 0.1
config.reward_scaling = 1
config.test_interval = 50
config.test_repetitions = 1
config.save_interval = config.episode_limit = 150
# config.logger = Logger('/Users/Morgans/Desktop/trading_system/log', gym.logger)
config.logger = Logger(root+'/log', gym.logger)
config.tag = tag
agent = DDPGAgent(config)
# agent


# from utils.misc import run_episodes
# agent.task._plot = agent.task._plot2 = None
# try:
#     run_episodes(agent)
# except KeyboardInterrupt as e:
#     save_ddpg(agent)
#     raise (e)
#


# plt.figure()
# df_online, df = load_stats_ddpg(agent)
# # sns.regplot(x="step", y="rewards", data=df_online, order=1)
# # plt.show()
# portfolio_return = (1+df_online.rewards.mean())
# returns = task.unwrapped.src.data[0,:,:1]
# ave_return = (1+returns).mean()
# print(ave_return, portfolio_return)
# agent.task.render('notebook')
# agent.task.render('humman')
# df_info = pd.DataFrame(agent.task.unwrapped.infos)
# df_info[["portfolio_value", "market_value"]].plot(title = "price", fig=plt.gcf())
# plt.show()

def test_algo(env, algo):
    # algo.config.task = task_fn_test()
    state = env.reset()
    done = False
    actions = []
    while not done:
        action = algo._step(state)
        actions.append(action)
        state, reward, done, info = env.step(action)
        if done:
            break
        # actions = getattr(action, 'value', action)
    df = pd.DataFrame(env.unwrapped.infos)
    df.index = pd.to_datetime(df['date'] * 1e9)
    env.render(mode = 'notebook')
    env.render(mode = 'humman')
    return df['portfolio_value'], df, actions
