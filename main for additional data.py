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
import matplotlib
import matplotlib.pyplot as plt
from utils.notebook_plot import LivePlotNotebook
matplotlib.rc('figure', figsize=[18, 10])


from windshow.ddpg_window_25 import agent as DDPGagent25
log_dir_ddpg_25 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDPGAgent-win25_weights.pth'
from windshow.ddpg_window_25 import test_algo, task_fn_test, task_fn_vali
DDPGagent25.worker_network.load_state_dict(torch.load(log_dir_ddpg_25))
portfolio_value_25, df_v_25, actions_25 = test_algo(task_fn_vali(), DDPGagent25)
portfolio_value_25, df_t_25, act25 = test_algo(task_fn_test(), DDPGagent25)



from windshow.ddpg_window_20 import agent as DDPGagent20
log_dir_ddpg_20 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDPGAgent-win10_20_weights_back2.pth'
# log_dir_ddpg_20 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDPGAgent-win20_weights.pth'
from windshow.ddpg_window_20 import test_algo, task_fn_test, task_fn_vali
DDPGagent20.worker_network.load_state_dict(torch.load(log_dir_ddpg_20))
portfolio_value_20, df_v_20, actions_20 = test_algo(task_fn_vali(), DDPGagent20)
portfolio_value_20, df_t_20, act20 = test_algo(task_fn_test(), DDPGagent20)



from windshow.ddpg_window_5 import agent as DDPGagent5
log_dir_ddpg_5 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDPGAgent-win5_weights.pth'
from windshow.ddpg_window_5 import test_algo, task_fn_test, task_fn_vali
DDPGagent5.worker_network.load_state_dict(torch.load(log_dir_ddpg_5))
portfolio_value_5, df_v_5, actions_5 = test_algo(task_fn_vali(), DDPGagent5)
portfolio_value_5, df_t_5, act5 = test_algo(task_fn_test(), DDPGagent5)


from windshow.ddpg_window_10 import agent as DDPGagent10
#'/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDDPGAgent-win10_alpha15_weights_back2.pth'
log_dir_ddpg_10 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDPGAgent-win10_weights.pth'
from windshow.ddpg_window_10 import test_algo, task_fn_test, task_fn_vali
DDPGagent10.worker_network.load_state_dict(torch.load(log_dir_ddpg_10))
portfolio_value_10, df_v_10, actions_10 = test_algo(task_fn_vali(), DDPGagent10)
portfolio_value_10, df_t_10, act10 = test_algo(task_fn_test(), DDPGagent10)



from windshow.Dddpg_window10_allpha50 import agent as Disagent1050
from windshow.Dddpg_window10_allpha50 import test_algo, task_fn_test, task_fn_vali
log_dir_dis_1050 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDDPGAgent-win10_alpha50_weights.pth'
Disagent1050.worker_network.load_state_dict(torch.load(log_dir_dis_1050))
portfolio_value_dis1050, df_v_dis1050, actions_dis1050 = test_algo(task_fn_vali(), Disagent1050)
portfolio_value_dis1050, df_t_dis1050, act_dis1050 = test_algo(task_fn_test(), Disagent1050)

from windshow.Dddpg_window10_alpha15 import agent as Disagent1015
from windshow.Dddpg_window10_alpha15 import test_algo, task_fn_test, task_fn_vali
log_dir_dis_1015 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDDPGAgent-win10_alpha15_weights.pth'
portfolio_value_dis1015, df_v_dis1015, actions_dis1015 = test_algo(task_fn_vali(), Disagent1015)
portfolio_value_dis1015, df_t_dis1015, act_dis1015 = test_algo(task_fn_test(), Disagent1015)


from windshow.Dddpg_window10_alpha30 import agent as Disagent1030
from windshow.Dddpg_window10_alpha30 import test_algo, task_fn_test, task_fn_vali
log_dir_dis_1030 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDDPGAgent-win10_alpha30_weights.pth'
portfolio_value_dis1030, df_v_dis1030, actions_dis1030 = test_algo(task_fn_vali(), Disagent1030)
portfolio_value_dis1030, df_t_dis1030, act_dis1030 = test_algo(task_fn_test(), Disagent1030)


from windshow.Dddpg_window10_alpha5 import agent as Disagent1005
from windshow.Dddpg_window10_alpha05 import test_algo, task_fn_test, task_fn_vali
log_dir_dis_1005 = '/Users/Morgans/Desktop/trading_system/video/addtional data weight/DDDPGAgent-win10_alpha5_weights.pth'
portfolio_value_dis1005, df_v_dis1005, actions_dis1005 = test_algo(task_fn_vali(), Disagent1005)
portfolio_value_dis1005, df_t_dis1005, act_dis1005 = test_algo(task_fn_test(), Disagent1005)




