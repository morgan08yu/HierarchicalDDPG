from window_length.ddpg_win_5 import agent as agent5
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from network.base_network import BasicNet
import logging
#from agent import ProximalPolicyOptimization, DisjointActorCriticNetdnet
from component import HighDimActionReplay, OrnsteinUhlenbeckProcess, AdaptiveParamNoiseSpec, hard_update, ddpg_distance_metric
from utils.config import Config
from utils.tf_logger import Logger
import gym
#import torchdp
from utils.normalizer import Normalizer, StaticNormalizer
import matplotlib
import matplotlib.pyplot as plt
from utils.notebook_plot import LivePlotNotebook

log_dir_5 = '/Users/Morgans/Desktop/trading_system/video/DDPGAgent-ddpg-cnn-agent-ETF-win7-etf_chn2.pth'
agent5.worker_network.load_state_dict(torch.load(log_dir_5))
from window_length.ddpg_win_5 import test_algo, task_fn_test, task_fn_vali
portfolio_value_51, df_v_5, actions_5 = test_algo(task_fn_vali(), agent5)
portfolio_value_52, df_t_5, act5 = test_algo(task_fn_test(), agent5)

from window_length.ddpg_win_10 import agent as agent10
from window_length.ddpg_win_10 import test_algo, task_fn_test, task_fn_vali
log_dir_10 ='/Users/Morgans/Desktop/trading_system/video/ETF weights/DDPGAgent-ddpg-cnn-agent-ETF-win10-etf_chn2.pth'
# agent10 =agent
agent10.worker_network.load_state_dict(torch.load(log_dir_10))
portfolio_value_101, df_v_10, actions_10 = test_algo(task_fn_vali(), agent10)
portfolio_value_102, df_t_10, act10 = test_algo(task_fn_test(), agent10)

matplotlib.rc('figure', figsize=[18, 10])
x = df_v_10.index
color=['blue', 'orange']
plot5 = LivePlotNotebook(log_dir=None, labels=['portfolio value of DDPG']+
                                              ['market value'], title='performance of DDPG', colors=color, ylabel= 'Portfolio value')
a5 = [df_v_10['portfolio_value']]
a15 = [df_v_10['market_value']]
plot5.update(x, a5+a15)
plt.show()



from window_length.ddpg_win_20 import agent as agent20
from window_length.ddpg_win_20 import test_algo, task_fn_test, task_fn_vali
log_dir_20 ='/Users/Morgans/Desktop/trading_system/video/DDPGAgent-ddpg-cnn-agent-ETF-win20-etf_chn2.pth'
# agent10 =agent
agent20.worker_network.load_state_dict(torch.load(log_dir_20))
portfolio_value_201, df_v_20, actions_20 = test_algo(task_fn_vali(), agent20)
portfolio_value_202, df_t_20, act20 = test_algo(task_fn_test(), agent20)

from window_length.ddpg_win_25 import agent as agent25
from window_length.ddpg_win_25 import test_algo, task_fn_test, task_fn_vali
log_dir_25 ='/Users/Morgans/Desktop/trading_system/video/DDPGAgent-ddpg-cnn-agent-ETF-win25-etf_chn2.pth'
# agent10 =agent
agent25.worker_network.load_state_dict(torch.load(log_dir_25))
portfolio_value_301, df_v_25, actions_25 = test_algo(task_fn_vali(), agent25)
portfolio_value_302, df_t_25, act25 = test_algo(task_fn_test(), agent25)

# from window_length.ddpg_win_40 import agent as agent40
# from window_length.ddpg_win_40 import test_algo, task_fn_test, task_fn_vali
# log_dir_40 ='/Users/Morgans/Desktop/trading_system/video/ETF weights/DDPGAgent-ddpg-cnn-agent-ETF-win40-etf_chn2.pth'
# # agent10 =agent
# agent10.worker_network.load_state_dict(torch.load(log_dir_40))
# portfolio_value_401, df_v_40, actions_40 = test_algo(task_fn_vali(), agent40)
# portfolio_value_402, df_t_40, act40 = test_algo(task_fn_test(), agent40)
import matplotlib
import matplotlib.pyplot as plt
from utils.notebook_plot import LivePlotNotebook
matplotlib.rc('figure', figsize=[18, 10])
color=['blue', 'red', 'green','purple' ,'orange']
x = df_v_10.index
plot5 = LivePlotNotebook(log_dir=None, labels=['DDPG with window size 5']+
                                              ['DDPG with window size 10']+
                                              ['DDPG with window size 20']+
                                              ['DDPG with window size 25']+
                                              ['Market value'], title='performance with different window size', colors =color, ylabel= 'Portfolio value')
win5 = [df_v_5['portfolio_value']]
win10 = [df_v_10['portfolio_value']]
win20=[df_v_20['portfolio_value']]
win25=[df_v_25['portfolio_value']]
market =[df_v_5['market_value']]
plot5.update(x, win5+win10+win20+win25+ market)
plt.show()



xx = df_t_10.index
plot = LivePlotNotebook(log_dir=None, labels=['DDPG with window size 5']+
                                             ['DDPG with window size 10']+
                                             ['DDPG with window size 20']+
                                             ['DDPG with window size 25']+
                                             ['Market value'], title='performance with different window size', colors =color, ylabel= 'Portfolio value')
win5_t = [df_t_5['portfolio_value']]
win10_t = [df_t_10['portfolio_value']]
win20_t = [df_t_20['portfolio_value']]
win25_t = [df_t_25['portfolio_value']]
market_t =[df_t_5['market_value']]
plot.update(xx, win5_t + win10_t +win20_t+ win25_t+market_t)
plt.show()


# from window_length.distributional_ddpg_win10 import agent as Dagent10
# from window_length.distributional_ddpg_win10 import test_algo, task_fn_test, task_fn_vali
# log_dir_D10 ='/Users/Morgans/Desktop/trading_system/video/distributional_ddpg_cvar_win10_etf.pth'
# # agent10 =agent
# Dagent10.worker_network.load_state_dict(torch.load(log_dir_D10))
# portfolio_value_d101, df_v_d10, actions_d10 = test_algo(task_fn_vali(), Dagent10)
# portfolio_value_d102, df_t_d10, act1d0 = test_algo(task_fn_test(), Dagent10)

from window_length.distribbution_ddpg_30 import agent as Dagent1030
from window_length.distribbution_ddpg_30 import test_algo, task_fn_test, task_fn_vali
log_dir_D1030 = '/Users/Morgans/Desktop/trading_system/video/distributional_ddpg_cvar_win10_301_etf.pth'
# agent10 =agent
Dagent1030.worker_network.load_state_dict(torch.load(log_dir_D1030))
portfolio_value_d10130, df_v_d1030, actions_d1030 = test_algo(task_fn_vali(), Dagent1030)
portfolio_value_d10230, df_t_d1030, act1d030 = test_algo(task_fn_test(), Dagent1030)

from window_length.Hierarchical_ddpg_win10 import agent2 as Hagent10
from window_length.Hierarchical_ddpg_win10 import test_performance, task_fn_test_H, task_fn_vali_H, task_fn_test_H7, task_fn_test_H13
log_dir_H10 = '/Users/Morgans/Desktop/trading_system/video/ETF weights/HiAgent-ddpg_cvar_win10_etf.pth'
Hagent10.worker_network_H.load_state_dict(torch.load(log_dir_H10))
portfolio_value_h101, df_v_h10, actions_h10 = test_performance(task_fn_vali_H(), Hagent10)
portfolio_value_h102, df_t_h10, act1h0 = test_performance(task_fn_test_H(), Hagent10)
portfolio_value_h1027, df_t_h107, act1h07 = test_performance(task_fn_test_H7(), Hagent10)
portfolio_value_h10213, df_t_h1013, act1h013 = test_performance(task_fn_test_H13(), Hagent10)
color=['royalblue', 'purple', 'slategray']
matplotlib.rc('figure', figsize=[18, 10])
plot = LivePlotNotebook(title='performance of HDDPG', labels=['C=5%']+['C=8%']+['C=13%'],colors=color, ylabel='Portfolio value')
x = df_t_h107.index
y_p = [df_t_h10['portfolio_value']]
y_p1 = [df_t_h107['portfolio_value']]
y_p2 = [df_t_h1013['portfolio_value']]
plot.update(x, y_p+y_p1+y_p2)
plt.show()





all_assets =['Cash']+ task_fn_test_H().env.sim.asset_names
matplotlib.rc('figure', figsize=[18, 10])
c = ['black','red', 'royalblue', 'purple', 'slategray','green']
# colors = [None] * len(all_assets) + ['green']
plot = LivePlotNotebook(title='prices & performance', labels=all_assets + ["portfolio of HDDPG"], colors=c,ylabel='value')
x = df_t_h10.index
y_portfolio = df_t_h10["portfolio_value"]
y_assets = [df_t_h10['price_' + name].cumprod() for name in all_assets]
plot.update(x, y_assets + [y_portfolio])
plt.show()

plot2 = LivePlotNotebook(labels=all_assets, title='optimal weights of HDDPG', ylabel='weight',colors=c)
ys = [df_t_h10['weight_' + name] for name in all_assets]
plot2.update(x, ys)
plt.show()

matplotlib.rc('figure', figsize=[18, 10])
plot3 = LivePlotNotebook( labels=['cost of HDDPG']+['cost of DDPG']+['cost of DDDPG'], title='transaction costs', ylabel='cost',colors=['green', 'blue','red'])
ys = [df_t_h10['cost'].cumsum()]
cost = [df_t_h10['cost_DDPG'].cumsum()]
cost_d=[df_t_d1030['cost'].cumsum()]
plot3.update(x, ys + cost+cost_d)
plt.show()

plot4 = LivePlotNotebook(labels=['CVaR_HDDPG']+['CVaR_DDPG']+['CVaR_DDDPG'], ylabel= 'CVaR', colors=['purple', 'slategray', 'royalblue'])
ys = [df_t_h10['CVaR']]
CVaR_DDPG = [df_t_h10['CVaR_DDPG']]
cvar_d=[sk_jul['CVaR']]
plot4.update(x, ys + CVaR_DDPG+cvar_d)
plt.show()



from window_length.ddpg_win_10 import agent as agent10
from window_length.ddpg_win_10 import test_algo, task_fn_test, task_fn_vali
log_dir_10 ='/Users/Morgans/Desktop/trading_system/video/ETF weights/DDPGAgent-ddpg-cnn-agent-ETF-win10-etf_chn2.pth'
# agent10 =agent
agent10.worker_network.load_state_dict(torch.load(log_dir_10))
portfolio_value_101, df_v_10, actions_10 = test_algo(task_fn_vali(), agent10)
portfolio_value_102, df_t_10, act10 = test_algo(task_fn_test(), agent10)


import matplotlib
import matplotlib.pyplot as plt
from utils.notebook_plot import LivePlotNotebook
matplotlib.rc('figure', figsize=[18, 10])
color =['blue','red', 'green', 'orange']
x = df_t_10.index
plot5 = LivePlotNotebook(log_dir=None, labels=['DDPG']+
                                              ['Distributional DDPG']+
                                              ['Hierarchical DDPG']+
                                              ['Market Value'], title='performance', colors=color,ylabel= 'Portfolio value')
ddpg = [df_t_10['portfolio_value']]
distributional=[df_t_d1030['portfolio_value']]
Heirarchical=[df_t_h10['portfolio_value']]
market =[df_t_10['market_value']]
plot5.update(x, ddpg+distributional+Heirarchical+market)
plt.show()


from window_length.distributional_ddpg_win10 import agent as Dagent1015
from window_length.distributional_ddpg_win10 import test_algo, task_fn_test, task_fn_vali
log_dir_D1015 ='/Users/Morgans/Desktop/trading_system/video/distributional_ddpg_cvar_win10_15_etf.pth'
# agent10 =agent
Dagent1015.worker_network.load_state_dict(torch.load(log_dir_D1015))
portfolio_value_d10115, df_v_d1015, actions_d1015 = test_algo(task_fn_vali(), Dagent1015)
portfolio_value_d10215, df_t_d1015, act1d015 = test_algo(task_fn_test(), Dagent1015)

# '/Users/Morgans/Desktop/trading_system/video/distributional_ddpg_cvar_win10_30_etf.pth'

from window_length.distributional_ddpg_win10 import agent as Dagent1050
from window_length.distributional_ddpg_win10 import test_algo, task_fn_test, task_fn_vali
log_dir_D1050 = '/Users/Morgans/Desktop/trading_system/video/distributional_ddpg_cvar_win1050_etf.pth'
Dagent1050.worker_network.load_state_dict(torch.load(log_dir_D1050))
portfolio_value_d10150, df_v_d1050, actions_d1050 = test_algo(task_fn_vali(), Dagent1050)
portfolio_value_d10250, df_t_d1050, act1d050 = test_algo(task_fn_test(), Dagent1050)

from window_length.distribution_ddpg05 import agent as Dagent1005
from window_length.distribution_ddpg05 import test_algo, task_fn_test, task_fn_vali
log_dir_D1005 ='/Users/Morgans/Desktop/trading_system/video/distributional_ddpg_cvar_win10_05_etf.pth'
# log_dir_H = '/Users/Morgans/Desktop/trading_system/video/distributional_ddpg_cvar_win10_05_etf.pth'
Dagent1005.worker_network.load_state_dict(torch.load(log_dir_D1005))
portfolio_value_d10105, df_v_d1005, actions_d1005 = test_algo(task_fn_vali(), Dagent1005)
portfolio_value_d10205, df_t_d1005, act1d005 = test_algo(task_fn_test(), Dagent1005)



from window_length.distribbution_ddpg_30 import agent as Dagent1030
from window_length.distribbution_ddpg_30 import test_algo, task_fn_test, task_fn_vali
log_dir_D1030 = '/Users/Morgans/Desktop/trading_system/video/distributional_ddpg_cvar_win10_301_etf.pth'
# agent10 =agent
Dagent1030.worker_network.load_state_dict(torch.load(log_dir_D1030))
portfolio_value_d10130, df_v_d1030, actions_d1030 = test_algo(task_fn_vali(), Dagent1030)
portfolio_value_d10230, df_t_d1030, act1d030 = test_algo(task_fn_test(), Dagent1030)

matplotlib.rc('figure', figsize=[22, 13])
matplotlib.rc('figure', figsize=[18, 10])
x = df_t_d1030.index
plot5 = LivePlotNotebook(log_dir=None, labels=['alpha =5%']+
                                              ['alpha =15%']+
                                              ['alpha =30%']+
                                              ['alpha =50%'], title='performance of Distributional DDPG',colors=color, ylabel= 'Portfolio value')
a5 = [df_t_d1005['portfolio_value']]
a15 = [df_t_d1015['portfolio_value']]
a30 = [df_t_d1030['portfolio_value']]
a50 = [df_t_d1050['portfolio_value']]
plot5.update(x, a5+a15+a30+a50)
plt.show()