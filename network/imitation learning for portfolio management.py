from __future__ import print_function
import os
os.environ["KERAS_BACKEND"] = "theano"
import numpy as np
# from utils.data import read_stock_history, index_to_date, date_to_index, normalize
import matplotlib.pyplot as plt
import matplotlib
# for compatible with python 3
import seaborn as sns
import bokeh.io
#import bokeh.mpl
import bokeh.plotting
import csv
import datetime
import numpy as np
import h5py
import pandas as pd
matplotlib.rcParams['figure.figsize'] = (10, 6)
plt.rc('legend', fontsize=20)
matplotlib.rcParams['figure.figsize'] = (10, 6)
plt.rc('legend', fontsize=20)
# JB's favorite Seaborn settings for notebooks
rc = {'lines.linewidth': 2,
      'axes.labelsize': 18,
      'axes.titlesize': 18,
      'axes.facecolor': 'DFDFE5'}
sns.set_context('notebook', rc=rc)
sns.set_style('darkgrid', rc=rc)
bokeh.io.output_notebook()

def read_stock_history(filepath='datasets/stocks_history.h5'):
    """ Read data from extracted h5
    Args:
        filepath: path of file
    Returns:
        history:
        abbreviation:
    """
    with h5py.File(filepath, 'r') as f:
        history = f['history'][:]
        abbreviation = f['abbreviation'][:].tolist()
        abbreviation = [abbr.decode('utf-8') for abbr in abbreviation]
    return history, abbreviation


def normalize(x):
    """ Create a universal normalization function across close/open ratio
    Args:
        x: input of any shape
    Returns: normalized data
    """
    return (x - 1) * 100

root = os.getcwd()
path_data = '~/Desktop/trading_system/HFT_data/financial_crisis/poloniex_fc.hf'
#path_data = root+'/HFT_data/four_stocks_includ/poloniex_fc.hf'
df_train = pd.read_hdf(path_data, key='train', encoding='utf-8')
df_test = pd.read_hdf(path_data, key='test', encoding='utf-8')

from Environment.Env_with_target import PortfolioEnv as Env
from utils.util import MDD, sharpe, softmax
from wrappers import SoftmaxActions, TransposeHistory, ConcatStates
from wrappers.logit import LogitActions
# df = DataSrc(df_train, 256, scale=True, scale_extra_cols=True, augment=0.00, window_length=50, random_reset=True)

import gym

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
    env = Env(df=df_train, steps=2100, window_length=10, output_mode='EIIE', trading_cost=0,
                       utility = 'Log', scale=True, scale_extra_cols=True, include_cash=True,
                       random_reset = False)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    # env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_test():
    env = Env(df=df_test, steps=650, window_length=10, output_mode='EIIE', trading_cost=0,
                       utility = 'Log', scale=True, scale_extra_cols=True, include_cash=True,
                       random_reset = False)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    # env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env
# from Environment.Env_with_target import DataSrc
# df = DataSrc(df_train, 1900, scale=False, scale_extra_cols=False, augment=0.00, window_length=10, random_reset=False, include_cash=True)

def create_dataset(env):
    task = env
    action = task.sim.w0
    steps = task.src.steps
    xs = []
    ys = []
    cash = np.ones((1,1))
    for i in range(steps-1):
        next_state, _, _, info = task.step(action)
        # x = next_state[:, 1:, :] # remove the weights
        x = next_state
        xs.append(x)
        labels = info['labels']
        truth_labels = labels[:, :, 0] / labels[:, :, 3]
        lb = np.concatenate((cash, truth_labels))
        lb = np.transpose(lb, (1, 0))
        y = np.argmax(lb) # include cash
        ys.append(y)
    return xs, ys

xs, ys = create_dataset(env = task_fn())

xs_test, ys_test = create_dataset(env = task_fn_test())
# change to array
state_dim = task_fn().state_dim
action_dim = task_fn().action_dim
xs = np.array(xs)
ys = np.array(ys)

from network.CNN import TorchCNN
from sklearn.preprocessing import label_binarize
from torch.autograd import Variable
import torch.nn.functional as F

class1 = [0,1,2,3,4]
y_label = label_binarize(ys, classes=class1)
import torch.nn as nn
import torch
net = TorchCNN(state_dim, action_dim, batch_norm=64, non_linear=F.relu)
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
# criterion = nn.MSELoss()
criterion = nn.NLLLoss()



from torch.utils.data import TensorDataset, Dataset, DataLoader

X_train = torch.tensor(xs)
Y_tain = torch.tensor(y_label)

data_train = TensorDataset(X_train, Y_tain)
train_loader = DataLoader(data_train, batch_size = 64, shuffle=True)
images, labels = next(iter(train_loader)) # check the one batch



num_epoches = 500
for epoch in range(num_epoches):
    print('epoch{}'.format(epoch+1))
    print('*'*10)
    running_loss = 0.0
    running_acc = 0.0
    for i, data in enumerate(train_loader, 1):
        img, label = data
        if torch.cuda.is_available():
            img = Variable(img).cuda()
            label = Variable(label).cuda()
        else:
            img = Variable(img).float()
            label = Variable(label).float()
        out = net.forward(img)
        # loss = nn.NLLLoss()(torch.log(out), label)
        loss = (-out.log() * label).sum(dim=1).mean()
        running_loss += loss.item() * label.size(0)
        _, pred = torch.max(out, 1)
        _, label_mac = torch.max(label, 1)
        num_correct = (pred == label_mac).sum()
        accuracy = (pred == label_mac).float().mean()
        # accuracy = accuracy(out, label)
        running_acc += num_correct.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print('Finish {} epoch,Loss:{:.6f},Acc:{:.6f}'.format(
        epoch+1, running_loss, running_acc/len(xs)))


    # net.eval()
    # eval_loss =0
    # eval_acc = 0
    # for i,data in enumerate(test_loader,1):
    #     img, label = data
    #     #判断是否可以使用GPU，若可以则将数据转化为GPU可以处理的格式。
    #     if torch.cuda.is_available():
    #         img = Variable(img).cuda()
    #         label = Variable(label).cuda()
    #     else:
    #         img = Variable(img)
    #         label = Variable(label)
    #
    #     out = net(img)
    #     loss = criterion(out,label)
    #     eval_loss += loss.item() * label.size(0)
    #     _, pred = torch.max(out,1)
    #     num_correct = (pred == label).sum()
    #     accuracy = (pred == label).float().mean()
    #     eval_acc += num_correct.item()
    #
    # print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
    #     test_dataset)), eval_acc/len(test_dataset)))
    # print()

    def save( file_name):
        with open(file_name, 'wb') as f:
            torch.save(net.state_dict(), f)

log_dir = '/Users/Morgans/Desktop/trading_system/video/initial_staring_point1.pth'
# net.save(log_dir)
torch.save(net.state_dict(), log_dir)


def test_net_env(env, net):
    # algo.config.task = task_fn_test()
    state = env.reset()
    done = False
    actions = []
    while not done:
        action = net.forward(np.stack([state])).flatten()
        actions.append(action)
        state, reward, done, info = env.step(action.data.numpy())
        if done:
            break
        # actions = getattr(action, 'value', action)
    df = pd.DataFrame(env.unwrapped.infos)
    # df.index = pd.to_datetime(df['date'] * 1e9)
    env.render(mode='notebook')
    env.render(mode='humman')
    return df['portfolio_value'], df, actions

portfolio_value, df_v, actions = test_net_env(task_fn_test(), net)



# log_dir1 = '/Users/Morgans/Desktop/trading_system/video/imit_CNN%3A window = 7.h5'
# log_dir2 = '/Users/Morgans/Desktop/trading_system/video/imit_CNN%3A window = 14.h5'
# model_dic = torch.load(log_dir1)
#
# import h5py
# with h5py.File(log_dir1,'r') as f:
#     for fkey in f.keys():
#         print(f[fkey], fkey)
#
# from h5py import Dataset, Group, File
# with File(log_dir1,'r') as f:
# 	for k in f.keys():
# 		if isinstance(f[k], Dataset):
# 			print(f[k].value)
# 		else:
# 			print(f[k].name)