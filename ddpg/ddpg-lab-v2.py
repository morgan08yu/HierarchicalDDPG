import os

import matplotlib
# os.sys.path.append(os.path.abspath('.'))
# os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system/'))
from matplotlib import pyplot as plt

# matplotlib.use('nbAgg', force=True)
matplotlib.rc('figure', figsize=[15, 10])
# matplotlib.use('TkAgg')
import logging

logger = log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
logging.basicConfig()
log.info('%s logger started', __name__)
# from utils.data import read_stock_history, index_to_date, date_to_index, normalize
import pandas as pd

window = 10
root = os.getcwd()
steps = 256
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
from wrappers import SoftmaxActions, TransposeHistory, ConcatStates

# df_train = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/ten_stock/poloniex_ten_sh.hf', key='train')
# df_test = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/ten_stock/poloniex_ten_sh.hf', key='test')
path_data = root + '/HFT_data/financial_crisis/poloniex_fc.hf'
# path_data = root+'/HFT_data/four_stocks_includ/poloniex_fc.hf'
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
    env = PortfolioEnv(df=df_train, steps=500, window_length=window, output_mode='EIIE', trading_cost=0.0025,
                       utility='Log', scale=False, scale_extra_cols=False)
    #env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env


def task_fn_test():
    env = PortfolioEnv(df=df_test, steps=650, window_length = window, output_mode='EIIE', trading_cost=0.0025,
                       utility='Log', scale=False, scale_extra_cols=False)
    #env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env


def task_fn_vali():
    env = PortfolioEnv(df=df_train, steps=2000, window_length = window, output_mode='EIIE', trading_cost=0.0025,
                       utility='Log', scale=False, scale_extra_cols=False)
    #env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env


import pickle


def save_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = root + '/video/%s-%s-model-%s.bin' % (
        agent_type, config.tag, agent.task.name)
    agent.save(save_file)
    return save_file


def load_ddpg(agent):
    agent_type = agent.__class__.__name__
    save_file = root + '/video/%s-%s-model-%s.bin' % (
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
from agent import DisjointActorCriticNet
from component import HighDimActionReplay, OrnsteinUhlenbeckProcess, AdaptiveParamNoiseSpec
from utils.config import Config
from utils.tf_logger import Logger
import gym
from utils.normalizer import StaticNormalizer

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
        #self.state_normalizer = StaticNormalizer(self.task.state_dim)
        #self.reward_normalizer = StaticNormalizer(1)
        #self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.3, adaptation_coefficient=1)
        #self.actor_perturbed = self.worker_network.actor

    def soft_update(self, target, src):
        for target_param, param in zip(target.parameters(), src.parameters()):
            target_param.detach_()
            target_param.copy_(target_param * (1.0 - self.config.target_network_mix) +
                               param * self.config.target_network_mix)

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

            if video_recorder is not None:
                video_recorder.capture_frame()
            done = (done or (config.max_episode_length and steps >= config.max_episode_length))
            total_reward += reward

            # tensorboard logging
            prefix = 'test_' if deterministic else ''
            log_value(prefix + 'reward', reward, self.total_steps)
            #log_value(prefix + 'action', action, steps)
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
                grad_critic = nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip, norm_type=2)
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
                grad_actor = nn.utils.clip_grad_norm_(actor.parameters(), config.gradient_clip, norm_type=2)
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
        h0 = 4
        h1 = 8
        self.conv1 = nn.Conv2d(1, h0, (3, 1)) # TODO change input size from features to 1
        #nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.constant_(self.conv1.bias, 0.0)
        self.conv2 = nn.Conv2d(h0, h1, (stride_time, 1), stride=(stride_time, 1))
        #nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        #nn.init.constant_(self.conv2.bias, 0.0)
        self.conv3 = nn.Conv2d((h1), 1, (1, 1))
        nn.init.uniform_(self.conv3.weight, a = -0.003, b = 0.003)
        #nn.init.constant_(self.conv3.bias, 0.0)
        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d((h1+1))

        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, gpu, False)


    def forward (self, x):
        x = self.to_torch_variable(x)
        cash = torch.ones([x.shape[0], x.shape[1], x.shape[2], 1])
        x = torch.cat([cash, x], dim = 3)
        #w0 = x[:, :1, :1, :]  # weights from last step
        #x = x[:, :, 1:, :]
        w0 = x[:, :1, :1, :]  # weights from last step
        obs = x[:, 3, 1:, :]
        last_price = obs[:, -1, :]
        # obs = self.to_torch_variable(obs)[:, None, :,:]
        open_price = x[:, 1 :1:, :]
        y = x[:, 1, 1:, :] / x[:, 3, 1:, :]  #last_price[:, np.newaxis]
        y = (y - 1) * 100
        y = self.to_torch_variable(y)[:, None, :, :]

        phi0 = self.non_linear(self.conv1(y))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv2(phi0))
        #h = torch.cat([phi1, w0], 1)
        #if self.batch_norm:
            #h = self.bn2(h)
        h = phi1
        action = self.conv3(h)
        # add cash_bias before we softmax
        #cash_bias_int = 1
        #cash_bias = self.to_torch_variable(torch.ones(action.size())[:, :, :, :1] * cash_bias_int)
        #action = torch.cat([cash_bias, action], -1)
        batch_size = action.size()[0]
        action = action.view((batch_size, -1))
        if self.action_gate:
            action = self.action_scale * self.action_gate(action)
        action = F.softmax(action, dim = 1)
        #action = torch.mul(action, 1)
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
        h1 = 64
        self.action = actions = action_dim - 1
        self.conv1 = nn.Conv2d(1, h0, (3, 1)) #TODO change the input size from features to 1
        #nn.init.kaiming_uniform_(self.conv1.weight, mode='fan_in', nonlinearity='relu')
        self.conv2 = nn.Conv2d(h0, h1, (stride_time, 1), stride=(stride_time, 1))
        #nn.init.kaiming_uniform_(self.conv2.weight, mode='fan_in', nonlinearity='relu')
        self.layer3 = nn.Linear((h1 + 1) * (actions+1), 1)
        nn.init.uniform_(self.layer3.weight, 0, 0.01)
        #nn.init.constant_(self.layer3.bias, 0.0)

        #nn.init.kaiming_uniform_(self.layer3.weight,mode='fan_in', nonlinearity='softmax')
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d((h1+2))
        self.batch_norm = batch_norm

        BasicNet.__init__(self, None, gpu, False)

    def forward(self, x, action):
        x = self.to_torch_variable(x)
        action = self.to_torch_variable(action)[:, None, None, :]  # remove cash bias 1
        cash = torch.ones([ x.shape[0], x.shape[1], x.shape[2], 1])
        x = torch.cat([cash, x], dim= 3)
        #w0 = x[:, :1, :1, :]  # weights from last step
        #x = x[:, :, 1:, :]
        w0 = x[:, :1, :1, :]  # weights from last step
        obs = x[:, 3, 1:, :]
        open_price = x[:, 1, 1:, :]
        last_price = obs[:, -1, :]
        y = x[:, 1, 1:, :] / x[:, 3, 1:,:]   #last_price[:, np.newaxis] # TODO modify the opening /closing
        y = (y-1)*100
        y = self.to_torch_variable(y)[:, None, :, :]

        phi0 = self.non_linear(self.conv1(y))
        if self.batch_norm:
            phi0 = self.bn1(phi0)
        phi1 = self.non_linear(self.conv2(phi0))
        h = torch.cat([phi1, action], 1)  # remove last action
        if self.batch_norm:
            h = self.bn2(h)
        batch_size = x.size()[0]
        action = self.layer3(h.view((batch_size, -1)))
        return action

    def predict(self, x, action):
        return self.forward(x, action)




config = Config()
config.task_fn = task_fn
task = config.task_fn()
config.actor_network_fn = lambda: DeterministicActorNet(
    task.state_dim, task.action_dim, action_gate=None, action_scale=1.0, non_linear=F.relu,
    batch_norm=True, gpu=False)
config.critic_network_fn = lambda: DeterministicCriticNet(
    task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=False, gpu=False)
config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-4, weight_decay=3e-3)
config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-3, weight_decay=1e-3)
config.replay_fn = lambda: HighDimActionReplay(memory_size= int(1e4), batch_size=64)
config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.15, sigma=0.3, sigma_min=0.00002,
                                                            n_steps_annealing=10000)

# TODO  Try actor lr = 5e-5 1e-2 and critic 5e-4, 1e-3
config.discount = 0.99
config.min_memory_size = 5000
config.max_steps = 500*1000
config.max_episode_length = 3000
config.target_network_mix = 0.001
config.noise_decay_interval = 1000
config.gradient_clip = 20
config.min_epsilon = 0.1
config.reward_scaling = 1
config.test_interval = 10
config.test_repetitions = 1
config.save_interval = config.episode_limit = 50
# config.logger = Logger('/Users/Morgans/Desktop/trading_system/log', gym.logger)
config.logger = Logger(root + '/log', gym.logger)
config.tag = tag
agent = DDPGAgent(config)
#agent


from utils.misc import run_episodes

agent.task._plot = agent.task._plot2 = None
run_episodes(agent)
save_file = save_ddpg(agent)

# plt.figure()
# df_online, df = load_stats_ddpg(agent)
# sns.regplot(x="step", y="rewards", data=df_online, order=1)
# plt.show()
# portfolio_return = (1+df_online.rewards.mean())
# returns = task.unwrapped.src.data[0,:,:1]
# ave_return = (1+returns).mean()
# print(ave_return, portfolio_return)
agent.task.render('notebook')
agent.task.render('humman')
df_info = pd.DataFrame(agent.task.unwrapped.infos)
df_info[["portfolio_value", "market_value"]].plot(title="price", fig=plt.gcf())
plt.show()


def test_algo(env, algo):
    # algo.config.task = task_fn_test()
    state = env.reset()
    done = False
    actions = []
    while not done:
        action = algo._step(np.stack([state])).flatten()
        actions.append(action)
        state, reward, done, info = env.step(action)
        if done:
            break
        # actions = getattr(action, 'value', action)
    df = pd.DataFrame(env.unwrapped.infos)
    # df.index = pd.to_datetime(df['date'] * 1e9)
    env.render(mode='notebook')
    env.render(mode='humman')
    return df['portfolio_value'], df, actions


portfolio_value, df_v, actions = test_algo(task_fn_vali(), agent)

portfolio_value, df_v, actions = test_algo(task_fn_test(), agent)

# df_v[["portfolio_value", "market_value"]].plot(title = "price", fig=plt.gcf())

df_v[["portfolio_value", "market_value"]]
df_v['CVaR'].plot()
plt.show()

log_dir = '/Users/Morgans/Desktop/trading_system/video/DDPGAgent-ddpg-agent-noobs.pth'
agent.save(log_dir)


# torch.save(agent.worker_network.state_dict(), log_dir)
# torch.save(agent.worker_network, log_dir)
