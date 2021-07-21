import os
#os.sys.path.append(os.path.abspath('.'))
#os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system/'))
from matplotlib import pyplot as plt
import sys
import matplotlib
#matplotlib.use('nbAgg', force=True)
matplotlib.rc('figure', figsize=[15, 10])
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
import gym
from wrappers.logit import LogitActions
#df_train = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/ten_stock/poloniex_ten_sh.hf', key='train')
#df_test = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/ten_stock/poloniex_ten_sh.hf', key='test')
path_data = root+'/HFT_data/financial_crisis/poloniex_fc.hf'
df_train = pd.read_hdf(path_data, key='train', encoding='utf-8')
df_test = pd.read_hdf(path_data, key='test', encoding='utf-8')
#df_train = pd.read_hdf('/tmp/pycharm_project_927/HFT_data/financial_crisis/poloniex_fc.hf', key='train')
#df_test = pd.read_hdf('/tmp/pycharm_project_927/HFT_data/financial_crisis/poloniex_fc.hf', key='test')

class Preprocess(object):
    def __init__(self, df, window_length = 50, steps = 1800):
        self.steps = steps+1
        self.window_length = window_length
        self.idx = self.window_length
        self.asset_names = df.columns.levels[0].tolist()
        self.features = df.columns.levels[1].tolist()
        data = df.values.reshape((len(df), len(self.asset_names), len(self.features)))
        self._data = np.transpose(data, (1, 0, 2))  # _data =(asset_names, time, features)
        self._times = df.index
        self.price_columns = ['Close', 'High', 'Low', 'Open']
        self.non_price_columns = set(df.columns.levels[1]) - set(self.price_columns)
        if True:
            x = self._data.reshape((-1, len(self.features)))
            self.stats = dict(mean=x.mean(0), std=x.std(0))
        self.reset()

    def _step(self):
        # get history matrix from dataframe
        self.step += 1
        data_window = self.data[:, self.step:self.step + self.window_length,:].copy() #TODO
        #truth_obs = self._data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        cprice = self.data[:, self.step:self.step + self.window_length, 3]
        "price relative change for closing pricing "
        y1 = data_window[:, -1, 3] / data_window[:, -2, 3]  # features = (open, close, high, low, vol)
        y1 = np.concatenate([[1.0], y1])  # add cash price
        # y1 should be the ('cash', 'stock1', 'stock2','stock3',..)
        # (eq 18) X: prices are divided by close price
        nb_pc = len(self.price_columns)
        last_close_price = data_window[:, -1, 3]
        data_window[:, :, :nb_pc] /= last_close_price[:,np.newaxis, np.newaxis]
        #data_window[:, :, :nb_pc] = (data_window[:, :, :nb_pc] - 1)*100
        data_window[:, :, nb_pc:] -= self.stats["mean"][None, None, nb_pc:] #TODO
        data_window[:, :, nb_pc:] /= self.stats["std"][None, None, nb_pc:]  #TODO
        data_window[:, :, nb_pc:] = np.clip(data_window[:, :, nb_pc:],
                self.stats["mean"][nb_pc:] - self.stats["std"][nb_pc:] * 10,
                self.stats["mean"][nb_pc:] + self.stats["std"][nb_pc:] * 10)
        history = data_window #-1) * 100
        done = bool(self.step >= self.steps)
        return history, y1, done, cprice

    def reset(self):
        self.step = 0
        "extract data for this episode"
        self.idx = np.random.randint(low=self.window_length + 1, high=self._data.shape[1] - self.steps - 2)
        self.data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :].copy()
        self.times = self._times[self.idx - self.window_length:self.idx + self.steps + 1]

steps=1800
state_train = np.zeros(shape=(1800,4,50,5))
pp = Preprocess(df = df_train)
pp.reset()
for i in range(steps):
    state_train[i,] = pp._step()[0]
state_train = np.transpose(state_train, (0,3,2,1))
state_train_reshape = np.transpose(state_train,(0,2,1,3) ) #(1800, 50, 5, 4)
state_train_reshape = state_train_reshape.reshape(1800, 50, -1)

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, bidirectional=True):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=bidirectional)
    def forward(self, inputs):
        h0 = torch.randn(1, inputs.shape[1], self.hidden_size )
        c0 = torch.randn(1, inputs.shape[1], self.hidden_size )
        output, (hn, cn) = self.lstm(tensor(inputs),(h0,c0))
        #output, hidden = self.lstm(inputs, hidden)
        return output, (hn, cn)
    def init_hidden(self):
        return (torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size),
                torch.zeros(1 + int(self.bidirectional), 1, self.hidden_size))



class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, output_size): #, vocab_size):
        super(AttentionDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn = nn.Linear(hidden_size + output_size, 1)
        self.lstm = nn.LSTM(hidden_size, output_size)  # if we are using embedding hidden_size should be added with embedding of vocab size
        #self.final = nn.Linear(output_size, vocab_size)
    def init_hidden(self):
        return (torch.zeros(1, 1, self.output_size),
                torch.zeros(1, 1, self.output_size))
    def forward(self, decoder_hidden, encoder_outputs, input):
        weights = []
        for i in range(len(encoder_outputs)):
            print(decoder_hidden[0][0].shape)
            print(encoder_outputs[0].shape)
            weights.append(self.attn(torch.cat((decoder_hidden[0][0],encoder_outputs[i]), dim=1)))
        normalized_weights = F.softmax(torch.cat(weights, 1), 1)

        attn_applied = torch.bmm(normalized_weights.unsqueeze(1),
                                 encoder_outputs.view(1, -1, self.hidden_size))

        input_lstm = torch.cat((attn_applied[0], input[0]), dim=1)  # if we are using embedding, use embedding of input here instead

        output, hidden = self.lstm(input_lstm.unsqueeze(0), decoder_hidden)

        #output = self.final(output[0])

        return output, hidden, normalized_weights









import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, UpSampling1D, BatchNormalization, LSTM, RepeatVector, Flatten, TimeDistributed, Masking
from tensorflow.keras.models import Model
from tensorflow.keras.models import model_from_json
from tensorflow.keras import regularizers
import datetime
import time
import requests as req
import json
import pandas as pd
import pickle
import os


from tensorflow.keras import backend as K
from tensorflow.keras import regularizers, constraints, initializers, activations
from tensorflow.keras.layers import RNN   #Recurrent
from tensorflow.keras.layers import Layer, InputSpec

tfPrint = lambda d, T: tf.Print(input_=T, data=[T, tf.shape(T)], message=d)

def _time_distributed_dense(x, w, b=None, dropout=None,
                            input_dim=None, output_dim=None,
                            timesteps=None, training=None):
    """Apply `y . w + b` for every temporal slice y of x.
    # Arguments
        x: input tensor.
        w: weight matrix.
        b: optional bias vector.
        dropout: wether to apply dropout (same dropout mask
            for every temporal slice of the input).
        input_dim: integer; optional dimensionality of the input.
        output_dim: integer; optional dimensionality of the output.
        timesteps: integer; optional number of timesteps.
        training: training phase tensor or boolean.
    # Returns
        Output tensor.
    """
    if not input_dim:
        input_dim = K.shape(x)[2]
    if not timesteps:
        timesteps = K.shape(x)[1]
    if not output_dim:
        output_dim = K.shape(w)[1]

    if dropout is not None and 0. < dropout < 1.:
        # apply the same dropout pattern at every timestep
        ones = K.ones_like(K.reshape(x[:, 0, :], (-1, input_dim)))
        dropout_matrix = K.dropout(ones, dropout)
        expanded_dropout_matrix = K.repeat(dropout_matrix, timesteps)
        x = K.in_train_phase(x * expanded_dropout_matrix, x, training=training)

    # collapse time dimension and batch dimension together
    x = K.reshape(x, (-1, input_dim))
    x = K.dot(x, w)
    if b is not None:
        x = K.bias_add(x, b)
    # reshape to 3D tensor
    if K.backend() == 'tensorflow':
        x = K.reshape(x, K.stack([-1, timesteps, output_dim]))
        x.set_shape([None, None, output_dim])
    else:
        x = K.reshape(x, (-1, timesteps, output_dim))
    return x

class AttentionDecoder:
    def __init__(self, units, output_dim,
                 activation='tanh',
                 return_probabilities=False,
                 name='AttentionDecoder',
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        """
        Implements an AttentionDecoder that takes in a sequence encoded by an
        encoder and outputs the decoded states
        :param units: dimension of the hidden state and the attention matrices
        :param output_dim: the number of labels in the output space

        references:
            Bahdanau, Dzmitry, Kyunghyun Cho, and Yoshua Bengio.
            "Neural machine translation by jointly learning to align and translate."
            arXiv preprint arXiv:1409.0473 (2014).
        """
        self.units = units
        self.output_dim = output_dim
        self.return_probabilities = return_probabilities
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(AttentionDecoder, self).__init__(**kwargs)
        self.name = name
        self.return_sequences = True  # must return sequences

    def build(self, input_shape):
        """
          See Appendix 2 of Bahdanau 2014, arXiv:1409.0473
          for model details that correspond to the matrices here.
        """

        self.batch_size, self.timesteps, self.input_dim = input_shape

        if self.stateful:
            super(AttentionDecoder, self).reset_states()

        self.states = [None, None]  # y, s

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.W_a = self.add_weight(shape=(self.units, self.units),
                                   name='W_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the r (reset) gate
        """
        self.C_r = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_r = self.add_weight(shape=(self.units, self.units),
                                   name='U_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_r = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_r',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_r = self.add_weight(shape=(self.units, ),
                                   name='b_r',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        """
            Matrices for the z (update) gate
        """
        self.C_z = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_z = self.add_weight(shape=(self.units, self.units),
                                   name='U_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_z = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_z',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_z = self.add_weight(shape=(self.units, ),
                                   name='b_z',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for the proposal
        """
        self.C_p = self.add_weight(shape=(self.input_dim, self.units),
                                   name='C_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_p = self.add_weight(shape=(self.units, self.units),
                                   name='U_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_p = self.add_weight(shape=(self.output_dim, self.units),
                                   name='W_p',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_p = self.add_weight(shape=(self.units, ),
                                   name='b_p',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)
        """
            Matrices for making the final prediction vector
        """
        self.C_o = self.add_weight(shape=(self.input_dim, self.output_dim),
                                   name='C_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.U_o = self.add_weight(shape=(self.units, self.output_dim),
                                   name='U_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.W_o = self.add_weight(shape=(self.output_dim, self.output_dim),
                                   name='W_o',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)
        self.b_o = self.add_weight(shape=(self.output_dim, ),
                                   name='b_o',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint)

        # For creating the initial state:
        self.W_s = self.add_weight(shape=(self.input_dim, self.units),
                                   name='W_s',
                                   initializer=self.recurrent_initializer,
                                   regularizer=self.recurrent_regularizer,
                                   constraint=self.recurrent_constraint)

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim))]
        self.built = True

    def call(self, x):
        # store the whole sequence so we can "attend" to it at each timestep
        self.x_seq = x

        # apply the a dense layer over the time dimension of the sequence
        # do it here because it doesn't depend on any previous steps
        # thefore we can save computation time:
        self._uxpb = _time_distributed_dense(self.x_seq, self.U_a, b=self.b_a,
                                             input_dim=self.input_dim,
                                             timesteps=self.timesteps,
                                             output_dim=self.units)

        return super(AttentionDecoder, self).call(x)

    def get_initial_state(self, inputs):
        # apply the matrix on the first time step to get the initial s0.
        s0 = activations.tanh(K.dot(inputs[:, 0], self.W_s))

        # from keras.layers.recurrent to initialize a vector of (batchsize,
        # output_dim)
        y0 = K.zeros_like(inputs)  # (samples, timesteps, input_dims)
        y0 = K.sum(y0, axis=(1, 2))  # (samples, )
        y0 = K.expand_dims(y0)  # (samples, 1)
        y0 = K.tile(y0, [1, self.output_dim])

        return [y0, s0]

    def step(self, x, states):

        ytm, stm = states

        # repeat the hidden state to the length of the sequence
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_a)

        # calculate the attention probabilities
        # this relates how much other timesteps contributed to this one.
        et = K.dot(activations.tanh(_Wxstm + self._uxpb),
                   K.expand_dims(self.V_a))
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, self.x_seq, axes=1), axis=1)
        # ~~~> calculate new hidden state
        # first calculate the "r" gate:

        rt = activations.sigmoid(
            K.dot(ytm, self.W_r)
            + K.dot(stm, self.U_r)
            + K.dot(context, self.C_r)
            + self.b_r)

        # now calculate the "z" gate
        zt = activations.sigmoid(
            K.dot(ytm, self.W_z)
            + K.dot(stm, self.U_z)
            + K.dot(context, self.C_z)
            + self.b_z)

        # calculate the proposal hidden state:
        s_tp = activations.tanh(
            K.dot(ytm, self.W_p)
            + K.dot((rt * stm), self.U_p)
            + K.dot(context, self.C_p)
            + self.b_p)

        # new hidden state:
        st = (1-zt)*stm + zt * s_tp

        yt = activations.softmax(
            K.dot(ytm, self.W_o)
            + K.dot(stm, self.U_o)
            + K.dot(context, self.C_o)
            + self.b_o)

        if self.return_probabilities:
            return at, [yt, st]
        else:
            return yt, [yt, st]

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """
        if self.return_probabilities:
            return (None, self.timesteps, self.timesteps)
        else:
            return (None, self.timesteps, self.output_dim)

    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.output_dim,
            'units': self.units,
            'return_probabilities': self.return_probabilities
        }
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


encoding_dim = (50, 5*4)
window_length = 50
inputs = Input(shape=(50, 5*4))
encoded = LSTM(3)(inputs)
#encoded = Attention(4)(encoded)
decoded = RepeatVector(50)(encoded)
decoded = LSTM(20, return_sequences=True)(decoded)
sequence_autoencoder = Model(inputs, decoded)
sequence_autoencoder.summary()
sequence_autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
history = sequence_autoencoder.fit(state_train_reshape, state_train_reshape,
                epochs=128,
                batch_size=1024,
                shuffle=True,
                validation_data=(state_train_reshape, state_train_reshape))
#decoded_stocks = sequence_autoencoder.predict()
state_train_reshape


inputs = Input(shape=(50,5,4))
actinput = TimeDistributed(Flatten(input_shape=(5, 4)))(inputs)
encoded = LSTM(20)(actinput)
#decoded = AttentionDecoder(50,50)(encoded)
decoded = RepeatVector(50)(encoded)
decoded = LSTM(20, return_sequences=True, activation='tanh')(decoded)
#decoded = TimeDistributed(LSTM(20, return_sequences=True, activation='tanh'))(decoded)
sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
sequence_autoencoder.summary()
sequence_autoencoder.compile(optimizer='adam', loss='mean_absolute_error')
sequence_autoencoder.fit(state_train, state_train,
                epochs=128,
                batch_size=1024,
                shuffle=True,
                validation_data=(state_train, state_train))



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
    env = PortfolioEnv(df=df_train, steps=128, window_length=window, output_mode='EIIE',
                       utility = 'MV', scale= True, scale_extra_cols=True)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_test():
    env = PortfolioEnv(df=df_test, steps=620, window_length=window, output_mode='EIIE',
                       utility='MV', scale= True, scale_extra_cols=True)
    env = TransposeHistory(env)
    env = ConcatStates(env)
    #env = SoftmaxActions(env)
    env = DeepRLWrapper(env)
    return env

def task_fn_vali():
    env = PortfolioEnv(df=df_train, steps=2000, window_length=window, output_mode='EIIE',
                       utility='MV', scale= True, scale_extra_cols=True)
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
        self.state_normalizer = StaticNormalizer(self.task.state_dim)
        self.reward_normalizer = StaticNormalizer(1)
        self.param_noise = AdaptiveParamNoiseSpec(initial_stddev=0.1, desired_action_stddev=0.3, adaptation_coefficient = 1)
        self.actor_perturbed = self.worker_network.actor

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
        #state = self.state_normalizer(state)
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
                #action = np.clip(action, 0, 1)
                #else:
                    #action += max(self.epsilon, config.min_epsilon) * self.random_process.sample()
                    #self.epsilon -= self.d_epsilon
                #action += self.random_process.sample()
                #actor.train()
                #action = action.data
                #action += torch.Tensor(self.random_process.sample())
            next_state, reward, done, info = self.task.step(action)
            #next_state = self.state_normalizer(next_state)

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
            #reward = self.reward_normalizer(reward)
            if not deterministic:
                self.replay.feed([state, action, reward, next_state, int(done)])
                self.total_steps += 1

            steps += 1
            state = next_state

            if done:
                break

            if not deterministic and self.replay.size() >= config.min_memory_size:
                #self.worker_network.train()
                experiences = self.replay.sample()
                states, actions, rewards, next_states, terminals = experiences
                states = tensor(states)
                actions = tensor(actions)
                rewards = tensor(rewards).unsqueeze(-1)
                mask = tensor(1-terminals).unsqueeze(-1)
                next_states = tensor(next_states)
                q_next = target_critic.predict(next_states, target_actor.predict(next_states))
                q_next = config.discount * q_next * mask
                q_next.add_(rewards)
                q_next = q_next.detach()
                q = critic.predict(states, actions)
                critic_loss = self.criterion(q, q_next)
                #TD error
                #critic_loss = (q - q_next).pow(2).mul(0.5).sum(-1).mean() # critic_loss/ TD_error
                #  critic network updating
                critic.zero_grad()
                self.critic_opt.zero_grad()
                critic_loss.backward()
                grad_critic = nn.utils.clip_grad_norm_(critic.parameters(), config.gradient_clip)
                self.critic_opt.step()

                #  actor network updating
                Actions = actor.predict(states, False)
                #var_actions = Variable(actions.data, requires_grad=True)
                q = critic.predict(states, Actions)
                #q = critic.predict(states, var_actions)
                policy_loss = -q.mean()
                #q.backward(torch.ones(q.size()))
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
        h2 = 8
        h1 = 8
        self.conv0 = nn.Conv2d(features, h0, (3, 3), stride = (1,1), padding=(1,1)) # input 64*5 *50 *10 out 64* 48 *8
        self.conv1 = nn.Conv2d(h0, h2, (3, 1)) # input 64 * 50 * 10   output 64 *48 *8
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.conv3 = nn.Conv2d((h1 + 1), 1, (1, 1))
        self.out = nn.Linear(5, 5)

        self.action_scale = action_scale
        self.action_gate = action_gate
        self.non_linear = non_linear

        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1+1)

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

        phi0 = self.non_linear(self.conv0(x))
        if self.batch_norm:
            phi0 = self.bn1(phi0)

        phi1 = self.non_linear(self.conv1(phi0))
        if self.batch_norm:
            phi1 = self.bn2(phi1)
        phi2 = self.non_linear(self.conv2(phi1))
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
        action = self.non_linear(self.out(action))
       #action /= action.sum()
        #action = F.softmax(self.out(action), dim =1)
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
        h2 = 8
        h1 = 8
        self.action = actions = action_dim - 1
        self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1,1))
        self.conv1 = nn.Conv2d(h0, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        self.layer3 = nn.Linear((h1 + 2) * actions, 1)
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
        action = self.to_torch_variable(action)[:, None, None, :-1]  # remove cash bias

        w0 = x[:, :1, :1, :]  # weights from last step
        x = x[:, :, 1:, :]

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


config = Config()
config.task_fn = task_fn
task = config.task_fn()
config.actor_network_fn = lambda: DeterministicActorNet(
    task.state_dim, task.action_dim, action_gate=None, action_scale=1.0, non_linear=F.relu,
    batch_norm=True, gpu=False)
config.critic_network_fn = lambda: DeterministicCriticNet(
    task.state_dim, task.action_dim, non_linear=F.relu, batch_norm=True, gpu=False)
config.network_fn = lambda: DisjointActorCriticNet(config.actor_network_fn, config.critic_network_fn)
config.actor_optimizer_fn = lambda params: torch.optim.Adam(params, lr=5e-6)
config.critic_optimizer_fn = lambda params: torch.optim.Adam(params, lr=1e-5) #weight_decay=0.01)
#config.replay_fn = lambda: ReplayMemory(capacity=int(1e9))
config.replay_fn = lambda: HighDimActionReplay(memory_size=int(1e6), batch_size=64)
config.random_process_fn = lambda: OrnsteinUhlenbeckProcess(size=task.action_dim, theta=0.25, sigma=0.25, sigma_min=0.01, n_steps_annealing=10000)

config.discount = 0.9
config.min_memory_size = 10000
config.max_steps = 100000
config.max_episode_length = 3000
config.target_network_mix = 0.001
config.noise_decay_interval = 10000
config.gradient_clip = 20
config.min_epsilon = 0.1
config.reward_scaling = 1
config.test_interval = 50
config.test_repetitions = 1
config.save_interval = config.episode_limit = 250
#config.logger = Logger('/Users/Morgans/Desktop/trading_system/log', gym.logger)
config.logger = Logger(root+'/log', gym.logger)
config.tag = tag
agent = DDPGAgent(config)
#agent


from utils.misc import run_episodes
agent.task._plot = agent.task._plot2 = None
try:
    run_episodes(agent)
except KeyboardInterrupt as e:
    save_ddpg(agent)
    raise (e)



plt.figure()
df_online, df = load_stats_ddpg(agent)
#sns.regplot(x="step", y="rewards", data=df_online, order=1)
#plt.show()
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
    #algo.config.task = task_fn_test()
    state = env.reset()
    done = False
    actions = []
    while not done:
        action = algo._step(state)
        actions.append(action)
        state, reward, done, info = env.step(action)
        if done:
            break
        #actions = getattr(action, 'value', action)
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


df_v['CVaR'].plot()
plt.show()