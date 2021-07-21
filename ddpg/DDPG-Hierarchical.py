import os
#os.sys.path.append(os.path.abspath('.'))
#os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system/'))
from matplotlib import pyplot as plt
import sys
import matplotlib
#matplotlib.use('nbAgg', force=True)
matplotlib.rc('figure', figsize=[18, 10])
#matplotlib.use('TkAgg')
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
#from utils.data import read_stock_history, index_to_date, date_to_index, normalize
import matplotlib
import pandas as pd
from tqdm import tqdm_notebook as tqdm
from collections import Counter
import tempfile
import logging
import time
import seaborn as sns
from scipy.stats import norm
window = 50
root = os.getcwd()
steps = 128
import datetime
ts = datetime.datetime.utcnow().strftime('%Y%m%d_%H-%M-%S')
save_path = root+'/log_TEST'
#save_path
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

#from Environment.DDPGPEnv import PortfolioEnv
from Environment.ENV import PortfolioEnv
from utils.util import MDD, sharpe, softmax
from wrappers import RobSoftmaxActions, RobTransposeHistory, RobConcatStates

from wrappers.logit import LogitActions
#df_train = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/ten_stock/poloniex_ten_sh.hf', key='train')
#df_test = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/ten_stock/poloniex_ten_sh.hf', key='test')
path_data = root+'/HFT_data/financial_crisis/poloniex_fc.hf'
df_train = pd.read_hdf(path_data, key='train', encoding='utf-8')
df_test = pd.read_hdf(path_data, key='test', encoding='utf-8')
#df_train = pd.read_hdf('/tmp/pycharm_project_927/HFT_data/financial_crisis/poloniex_fc.hf', key='train')
#df_test = pd.read_hdf('/tmp/pycharm_project_927/HFT_data/financial_crisis/poloniex_fc.hf', key='test')


import gym

class DeepRLWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.render_on_reset = False

        self.state_dim = self.observation_space.shape
        self.action_dim = self.action_space.shape[0]

        self.name = 'ENV'
        #self.success_threshold = 2 #TODO

    def normalize_state(self, state):
        return state

    def step(self, action1, action2):
        state, reward_gt, reward_at, done, info, z = self.env.step(action1, action2)
        #reward *= 1000  # often reward scaling
        return state, reward_gt, reward_at, done, info, z

    def reset(self):
        # here's a roundabout way to get it to plot on reset
        if self.render_on_reset:
            self.env.render('notebook')
        return self.env.reset()

def task_fn():
    env = PortfolioEnv(df=df_train, steps=steps, window_length=window, output_mode='EIIE',gamma = 2, c= -0.7,
                       utility = 'Log', scale=True, scale_extra_cols=True)
    env = RobTransposeHistory(env)
    env = RobConcatStates(env)
    #env = RobSoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_test():
    env = PortfolioEnv(df=df_test, steps=620, window_length=window, output_mode='EIIE', gamma= 2, c= -0.7,
                       utility='Log', scale=True, scale_extra_cols=True)
    env = RobTransposeHistory(env)
    env = RobConcatStates(env)
    #env = RobSoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_vali():
    env = PortfolioEnv(df=df_train, steps=2000, window_length=window, output_mode='EIIE', gamma= 2, c=0.04,
                       utility='Log', scale=True, scale_extra_cols=True)
    env = RobTransposeHistory(env)
    env = RobConcatStates(env)
    #env = RobSoftmaxActions(env)
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
        steps, rewards_gt, rewards_at = pickle.load(open(online_stats_file, 'rb'))
    except FileNotFoundError:
        steps = []
        rewards_gt = []
        rewards_at=[]
    df_online = pd.DataFrame(np.array([steps, rewards_gt, rewards_at]).T, columns=['steps', 'rewards_gt','rewards_at'])
    if len(df_online):
        df_online['step'] = df_online['steps'].cumsum()
        df_online.index.name = 'episodes'
    stats_file = root + '/video/%s-%s-all-stats-%s.bin' % (
        agent_type, config.tag, agent.task.name)
    try:
        stats = pickle.load(open(stats_file, 'rb'))
    except FileNotFoundError:
        stats = {}
    df_g = pd.DataFrame(stats["test_rewards_gt"], columns=['rewards_gt'])
    df_a = pd.DataFrame(stats["test_rewards_at"], columns=['rewards_at'])
    if len(df_g):
        # df["steps"]=range(len(df))*50
        df_g.index.name = 'episodes'
    if len(df_a):
        df_a.index.name = 'episodes'
    return df_online, df_g, df_a


from sklearn.preprocessing import MinMaxScaler


def min_max_scale(data):
    w0 = data[:, :2, :]
    x = data[:, 2:, :]
    h, r, c = x.shape
    scar = np.zeros([h,r,c])
    scaler = MinMaxScaler()
    for h in range(h):
        scar[h] = scaler.fit_transform(x[h])
    scar = np.concatenate([w0, scar], 1)
    return scar

import logging
from agent import ProximalPolicyOptimization, DisjointActorCriticNet
from component import HighDimActionReplay, OrnsteinUhlenbeckProcess, AdaptiveParamNoiseSpec, hard_update, ddpg_distance_metric, RobHighDimActionReplay

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

class Agent(mp.Process):
    def __init__(self, config):
        self.config = config
        self.task = config.task_fn()
        self.worker_network = config.network_fn()
        self.target_network = config.network_fn()
        self.worker_network_H = config.network_fn_H()
        self.target_network_H = config.network_fn_H() # adding target network
        self.target_network.load_state_dict(self.worker_network.state_dict())
        self.target_network_H.load_state_dict(self.worker_network_H.state_dict())
        self.actor_opt = config.actor_optimizer_fn(self.worker_network.actor.parameters())
        self.critic_opt = config.critic_optimizer_fn(self.worker_network.critic.parameters())
        self.replay_DDPG = config.replay_fn()
        self.random_process = config.random_process_fn()
        self.criterion = nn.MSELoss()
        self.total_steps = 0
        self.alpha = 0.05
        self.replay_Hierachical = config.replay_Hierachical()
        self.actor_high_opt = config.actor_optimizer(self.worker_network_H.actor.parameters())
        self.critic_high_opt = config.critic_optimizer(self.worker_network_H.critic.parameters())

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) + param * self.config.target_network_mix)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def CVaR(self, cprice, gt): # TODO deal with unpreprocess cprice
        rows, cols = cprice.shape
        returns = np.empty([rows, cols - 1])
        for r in range(rows):
            for c in range(cols - 1):
                p0, p1 = cprice[r, c], cprice[r, c + 1]
                returns[r, c] = (p1 / p0) - 1  # TODO
        # calculate returns
        expreturns = np.array([])
        for r in range(rows):
            expreturns = np.append(expreturns, np.mean(returns[r]))
        # calculate covariances
        covars = np.cov(returns)
        expreturns_anu = (1 + expreturns) ** 250 - 1  # Annualize returns
        covars_anu = covars * 250 # Annualize variance
        mu_gt = np.dot(gt, expreturns)
        sigma_gt = np.sqrt(np.dot(gt.T, np.dot(covars, gt)))
        CVaR = self.alpha ** -1 * norm.pdf(norm.ppf(self.alpha)) * sigma_gt - mu_gt
        return CVaR

    def min_max_scale(self, data):
        w0 = data[:, :2, :]
        x = data[:, 2:, :]
        h, r, c = x.shape
        scar = np.zeros([h, r, c])
        scaler = MinMaxScaler()
        for h in range(h):
            scar[h] = scaler.fit_transform(x[h])
        scar = scar * 100
        scar = np.concatenate([w0, scar], 1)
        return scar

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
        actor_high = self.worker_network_H.actor
        critic_high = self.worker_network_H.critic
        target_actor_high = self.target_network_H.actor
        target_critic_high = self.target_network_H.critic
        steps = 0
        total_reward_gt = 0.0
        total_reward_at = 0.0
        while True:
            actor.eval()
            actor_high.eval()
            gt = actor.predict(np.stack([state])).flatten()
            if not deterministic:
                gt += self.random_process.sample()
            at = actor_high.predict(np.stack([state]), np.stack([gt])).flatten()
            if not deterministic:
                at += self.random_process.sample()
            next_state, reward_gt, reward_at, done, info, z = self.task.step(gt, at)
            # next_state = self.min_max_scale(next_state)
            if video_recorder is not None:
                video_recorder.capture_frame()
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))

            if z == 0:
                self.replay_DDPG.feed([state, gt, reward_gt, next_state, int(done)])
                total_reward_gt += reward_gt
                self.total_steps += 1
            else:
                self.replay_Hierachical.feed([state, gt, at, reward_at, next_state, int(done)])
                total_reward_at += reward_at
                self.total_steps += 1
            # total_reward_gt += reward_gt
            # total_reward_at += reward_at
            self.total_steps += 1

            # tensorboard logging
            prefix = 'test_' if deterministic else ''
            log_value(prefix + 'reward_gt', reward_gt, self.total_steps)
            log_value(prefix + 'reward_at', reward_at, self.total_steps)
            for key in info:
                log_value(key, info[key], self.total_steps)
            # self.replay_DDPG.feed([state, gt, reward_gt, next_state, int(done)])
            # self.replay_Hierachical.feed([state, gt, at, reward_at, next_state, int(done)])
            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.replay_DDPG.size() >= config.min_memory_size:
                self.worker_network.train()
                experiences1 = self.replay_DDPG.sample()
                states, actions, rewards_gt, next_states, terminals = experiences1
                states = tensor(states)
                actions = tensor(actions)
                rewards_gt = tensor(rewards_gt).unsqueeze(-1)
                mask = tensor(1-terminals).unsqueeze(-1)
                next_states = tensor(next_states)
                q_next = target_critic.predict(next_states, target_actor.predict(next_states))
                q_next = config.discount * q_next * mask
                q_next.add_(rewards_gt)
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
                Actions = actor.predict(states, False)  # TODO
                # var_actions = Variable(Actions.data, requires_grad=True)
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

            if not deterministic and self.replay_Hierachical.size() >= 500:
                self.worker_network_H.train()
                actor.eval()
                experiences2 = self.replay_Hierachical.sample()
                states_H, actions_gt, actions_at, rewards_at, next_states_H, terminals_H = experiences2
                states_H = tensor(states_H)
                actions_gt = tensor(actions_gt)
                actions_at = tensor(actions_at)
                rewards_at = tensor(rewards_at).unsqueeze(-1)
                mask_H = tensor(1-terminals_H).unsqueeze(-1)
                next_states_H = tensor(next_states_H)
                gt_next = actor.predict(next_states_H, False)
                gt_next = gt_next.detach()
                # gt_next = Variable(gt_next.data, requires_grad=False)
                q_next_H = target_critic_high.predict(next_states_H, gt_next, target_actor_high.predict(next_states_H, gt_next))
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
                # tensorboard logging #
                log_value('critic_loss_H', critic_loss_H.sum(), self.total_steps)
                log_value('policy_loss_H', policy_loss_H.sum(), self.total_steps)
                if config.gradient_clip:
                    log_value('grad_critic_H', grad_critic_H, self.total_steps)
                    log_value('grad_actor_H', grad_actor_H, self.total_steps)
                self.soft_update(self.target_network_H, self.worker_network_H)


        return total_reward_gt, total_reward_at, steps

    def _step(self, state):
        self.worker_network.eval()
        self.worker_network_H.eval()
        gt = self.worker_network.actor.predict(np.stack([state])).flatten()
        at = self.worker_network_H.actor.predict(np.stack([state]), np.stack([gt])).flatten()
        return gt, at


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
        h1 = 8
        self.conv0 = nn.Conv2d(features, h0, (3, 3), stride = (1, 1), padding=(1, 1)) # input 64*5 *50 *10 out 64* 48 *8
        self.conv1 = nn.Conv2d(h0, h2, (3, 1)) # input 64 * 50 * 10   output 64 *48 *8
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d((h1+1), 1, (1, 1))
        self.out = nn.Linear(5, 5)
        self.fc = nn.Dropout(0.1)

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1+1)
            #self.bn4 = nn.BatchNorm2d(h1)

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
        x = x[:, :, 1:, :]   # remove a_t

        phi0 = self.non_linear(self.conv0(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(phi0))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi1 = self.fc(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        phi2h = torch.cat([phi2, w0], 1)
        if self.batch_norm:
            phi2h = self.bn3(phi2h)
        action = self.conv3(phi2h)  # does not include cash account, add cash in next step.
        # add cash_bias before we softmax
        cash_bias_int = 1
        cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
        action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        if self.action_gate:
            action = self.action_scale * self.action_gate(action)
        action = self.non_linear(self.out(action))
       #action /= action.sum()
        #action = F.softmax(self.out(action), dim = 1)
        return action

    def predict(self, x, to_numpy=True):
        y = self.forward(x)
        if to_numpy:
            y = y.cpu().data.numpy()
        return y


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

        stride_time = state_dim[1] - 1 - 2 -1  #
        features = task.state_dim[0]
        h0 = 8
        h2 = 16
        h1 = 8
        self.conv0 = nn.Conv2d(features, h0, (3, 3), stride = (1,1), padding=(1,1)) # input 64*5 *50 *10 out 64* 48 *8
        self.conv1 = nn.Conv2d(h0, h1, (3, 1)) # input 64 * 50 * 10   output 64 *48 *8
        self.conv2 = nn.Conv2d(h1, h2, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d((h2 + 1), 1, (1, 1))
        self.out = nn.Linear(5, 5)
        self.fc = nn.Dropout(0.0)

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h1)
            self.bn3 = nn.BatchNorm2d(h2+1)
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
        x = x[:, :, 1:, :]   # remove a_t
        w1 = self.to_torch_variable(w1)[:, None, None, :-1]
        phi0 = self.non_linear(self.conv0(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(phi0))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        h = torch.cat([phi2, w1], 1)
        if self.batch_norm:
            h = self.bn3(h)
        action = self.conv3(h) # does not include cash account, add cash in next step.
        # add cash_bias before we softmax
        cash_bias_int = 1
        cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
        action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        if self.action_gate:
            action = self.action_scale * self.action_gate(action)
        action = self.non_linear(self.out(action))
        #action = F.softmax(self.out(action), dim =1)
        return action

    def predict(self, x, w1, to_numpy=True):
        y = self.forward(x, w1)
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
        h1 = 8
        self.action = actions = action_dim - 1
        self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1,1))
        self.conv1 = nn.Conv2d(h0, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer3 = nn.Linear((h1 + 2) * actions, 1)
        self.non_linear = non_linear
        self.fc = nn.Dropout(0.3)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1+2)
            #self.bn4 = nn.BatchNorm2d(h1)
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
        x = x[:, :, 1:, :] # TODO remove the action at

        phi0 = self.non_linear(self.conv0(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(phi0))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        h = torch.cat([phi2, w0, action], 1)
        if self.batch_norm:
            h = self.bn3(h)
        batch_size = x.size()[0]
        #action = self.non_linear(self.layer3(h))
        action = self.layer3(h.view((batch_size, -1)))
        return action

    def predict(self, x, action):
        return self.forward(x, action)


class Critichierachical(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 gpu=False,
                 batch_norm=False,
                 non_linear=F.relu):
        super(Critichierachical, self).__init__()
        stride_time = state_dim[1] - 1 - 2 - 1 #
        self.features = features = task.state_dim[0]
        h0 = 8
        h2 = 16
        h1 = 8
        self.action = actions = action_dim - 1
        self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1,1))
        self.conv1 = nn.Conv2d(h0, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer3 = nn.Linear((h1 + 2) * actions, 1) # TODO adding action2
        self.non_linear = non_linear
        self.fc = nn.Dropout(0.0)

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1+2)
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
        #w1 = x[:, :1, 1:2, :]# weights from last step
        x = x[:, :, 1:, :] # TODO remove the action at

        phi0 = self.non_linear(self.conv0(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv1(phi0))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
        h = torch.cat([phi2, action1, action2], 1)
        if self.batch_norm:
            h = self.bn3(h)
        batch_size = x.size()[0]
        #action = self.non_linear(self.layer3(h))
        action = self.layer3(h.view((batch_size, -1)))
        return action

    def predict(self, x, action1, action2):
        return self.forward(x, action1, action2)


config = Config()
config.task_fn = task_fn
task = config.task_fn()
config.actor_high = lambda: Hierachicalagent(
    task.state_dim, task.action_dim, action_gate=None, action_scale=1.0, non_linear=F.relu,
    batch_norm=True, gpu=False)
config.critic_high = lambda: Critichierachical(
    task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=True, gpu=False)
config.actor_network_fn = lambda: DeterministicActorNet(
    task.state_dim, task.action_dim, action_gate=None, action_scale=1.0, non_linear=F.relu,
    batch_norm=True, gpu=False)
config.critic_network_fn = lambda: DeterministicCriticNet(
    task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=True, gpu=False)
config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
config.network_fn_H = lambda: DisjointActorCriticNet(config.actor_high, config.critic_high)
config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-6) #weight_decay=1e-4)
config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=5e-6, weight_decay=1e-4)
config.actor_optimizer = lambda params: torch.optim.Adam(params, lr=1e-6) #weight_decay=1e-4)  #TODO tunning paarameters
config.critic_optimizer = lambda params: torch.optim.Adam(params, lr=5e-6, weight_decay=1e-4)
#weight_decay=0.01)
#config.replay_fn = lambda: ReplayMemory(capacity=int(1e9))
config.replay_fn = lambda: HighDimActionReplay(memory_size=int(1e6), batch_size=32)
config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.3, sigma=0.35, sigma_min=0.002, n_steps_annealing=10000)
config.replay_Hierachical = lambda: RobHighDimActionReplay(memory_size=int(1e6), batch_size=32)



config.discount = 0.90
config.min_memory_size = 15000
config.max_steps = 100000
config.max_episode_length = 3000
config.target_network_mix = 0.001
config.noise_decay_interval = 10000
config.gradient_clip = 10
config.min_epsilon = 0.1
config.reward_scaling = 1
config.test_interval = 50
config.test_repetitions = 1
config.save_interval = config.episode_limit = 150
#config.logger = Logger('/Users/Morgans/Desktop/trading_system/log', gym.logger)
config.logger = Logger(root+'/log', gym.logger)
config.tag = tag
agent = Agent(config)
#agent
log_dir = '/Users/Morgans/Desktop/trading_system/video/DDPGAgent-ddpg-agent.pth'
agent.worker_network.load_state_dict(torch.load(log_dir))
agent.target_network.load_state_dict(torch.load(log_dir))
from utils.misc import run_episodes, training
agent.task._plot = agent.task._plot2 = None
try:
    training(agent)
except KeyboardInterrupt as e:
    save_ddpg(agent)
    raise (e)


# check the plot

#plt.figure()
#df_online, df_g, df_a = load_stats_ddpg(agent)
#sns.regplot(x="step", y="rewards", data=df_online, order=1)
#plt.show()
#portfolio_return = (1+df_online.rewards.mean())
#returns = task.unwrapped.src.data[0,:,:1]
#ave_return = (1+returns).mean()
#print(ave_return, portfolio_return)
#agent.task.render('notebook')
#agent.task.render('humman')
#df_info = pd.DataFrame(agent.task.unwrapped.infos)
#df_info[["portfolio_value", "market_value"]].plot(title = "price", fig=plt.gcf())
#plt.show()

def test_performance(env, algo):
    #algo.config.task = task_fn_test()
    state = env.reset()
    done = False
    actions = []
    while not done:
        action1, action2 = algo._step(state)
        state, reward_gt, reward_at, done, info, z = env.step(action1, action2)
        if z == 0:
            actions.append(action1)
        else:
            actions.append(action2)
        if done:
            break
        #actions = getattr(action, 'value', action)
    df = pd.DataFrame(env.unwrapped.infos)
    df.index = pd.to_datetime(df['date'] * 1e9)
    env.render(mode = 'notebook')
    env.render(mode = 'humman')
    return df['portfolio_value'], df, actions


#portfolio_value, df_v, actions = test_algo(task_fn_vali(), agent)

portfolio_value, df_t, actions = test_performance(task_fn_test(), agent)

#df_v[["portfolio_value", "market_value"]].plot(title = "price", fig=plt.gcf())

df_t[["portfolio_value", "market_value"]].plot()
plt.show()
df_t.plot(y = ['CVaR_DDPG','CVaR'], use_index=True)
plt.show()
df_t.plot(y = ['Sharp ratio','Sharp ratio DDPG'], use_index=True)
plt.show()