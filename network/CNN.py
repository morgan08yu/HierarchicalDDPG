import os
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from network.base_network import BasicNet


class TorchCNN(nn.Module, BasicNet):
    def __init__(self,
                 state_dim,
                 action_dim,
                 batch_norm,
                 non_linear = F.relu):
        super(TorchCNN, self).__init__()
        stride_time = state_dim[1] - 1 - 2  #
        self.features = features = state_dim[0]
        h0 = 8
        h2 = 64
        h1 = 64
        self.action = actions = action_dim - 1
        self.conv0 = nn.Conv2d(features, h0, (3, 3), padding=(1, 1))
        self.conv1 = nn.Conv2d(h0, h2, (3, 1))
        self.conv2 = nn.Conv2d(h2, h1, (stride_time, 1), stride=(stride_time, 1))
        # self.conv3 = nn.Conv2d(h1, 1, (1, 1))
        self.layer3 = nn.Linear((h1) * action_dim, action_dim)
        # self.layer4 = nn.Linear(64, action_dim)

        self.non_linear = non_linear
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(h0)
            self.bn2 = nn.BatchNorm2d(h2)
            self.bn3 = nn.BatchNorm2d(h1)
            self.bn4 = nn.BatchNorm2d(64)
        self.batch_norm = batch_norm
        BasicNet.__init__(self, None, False, False)

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
        if self.batch_norm:
            phi2 = self.bn3(phi2)
        batch_size = phi2.size()[0]
        phi3 = self.layer3(phi2.view((batch_size, -1)))
        out = F.softmax(phi3, dim = 1)
        return out

    def predict(self, x, to_numpy = True):
        y = self.forward(x)
        if to_numpy == True:
            y = y.cpu().data.numpy()
        return y







from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import tensorflow as tf



class StockCNN:
    def __init__(self, nb_classes, window_length, weights_file='weights/cnn.h5'):
        self.model = None
        self.weights_file = weights_file
        self.nb_classes = nb_classes
        self.window_length = window_length

    def build_model(self, load_weights=True):
        """ Load training history from path
        Args:
            load_weights (Bool): True to resume training from file or just deploying.
                                 Otherwise, training from scratch.
        Returns:
        """
        if load_weights:
            self.model = load_model(self.weights_file)
            print('Successfully loaded model')
        else:
            self.model = Sequential()

            self.model.add(
                Conv2D(filters=32, kernel_size=(1, 3), input_shape=(self.nb_classes, self.window_length, 1),
                       activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Conv2D(filters=32, kernel_size=(1, self.window_length - 2), activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Flatten())
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(64, activation='relu'))
            self.model.add(Dropout(0.5))
            self.model.add(Dense(self.nb_classes, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy',
                               optimizer=Adam(lr=1e-3),
                               metrics=['accuracy'])
            print('Built model from scratch')
        self.model._make_predict_function()
        self.graph = tf.get_default_graph()

    def train(self, X_train, Y_train, X_val, Y_val, verbose=True):
        continue_train = True
        while continue_train:
            self.model.fit(X_train, Y_train, batch_size=128, epochs=10, validation_data=(X_val, Y_val),
                           shuffle=True, verbose=verbose)
            save_weights = input('Type True to save weights\n')
            if save_weights:
                self.model.save(self.weights_file)
            continue_train = input("True to continue train, otherwise stop training...\n")
        print('Finish.')

    def normalize(self, x):
        return (x-1)*100

    def evaluate(self, X_test, Y_test, verbose=False):
        return self.model.evaluate(X_test, Y_test, verbose=verbose)

    def predict(self, X_test, verbose=False):
        return self.model.predict(X_test, verbose=verbose)

    def predict_single(self, observation):
        """ Predict the action of a single observation
        Args:
            observation: (num_stocks + 1, window_length)
        Returns: a single action array with shape (num_stocks + 1,)
        """
        obsX = observation[:, -self.window_length:, 3:4] / observation[:, -self.window_length:, 0:1]
        obsX = self.normalize(obsX)
        obsX = np.expand_dims(obsX, axis=0)
        with self.graph.as_default():
            return np.squeeze(self.model.predict(obsX), axis=0)

####++++++++++++++++++++++++++++++++++++++++++++++++++++++


import numpy as np

from tensorflow.keras.layers import Dense, Activation, BatchNormalization, Dropout, Conv1D, Flatten, MaxPooling1D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import np_utils
from tensorflow.keras import utils as np_utils

def create_network_given_future(nb_classes, weight_path='weights/optimal_3_stocks.h5'):
    model = Sequential()
    model.add(Dense(512, input_shape=(nb_classes,)))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    try:
        model.load_weights(weight_path)
        print('Model load successfully')
    except:
        print('Build model from scratch')
    return model

def normalize(x):
    return (x-1)*100

def create_optimal_imitation_dataset(history, training_data_ratio=0.8, is_normalize=True):
    """ Create dataset for imitation optimal action given future observations
    Args:
        history: size of (num_stocks, T, num_features) contains (open, high, low, close)
        training_data_ratio: the ratio of training data
    Returns: un-normalized close/open ratio with size (T, num_stocks), labels: (T,)
             split the data according to training_data_ratio
    """
    num_stocks, T, num_features = history.shape
    cash_history = np.ones((1, T, num_features))
    history = np.concatenate((cash_history, history), axis=0)
    close_open_ratio = np.transpose(history[:, :, 3] / history[:, :, 0])
    if is_normalize:
        close_open_ratio = normalize(close_open_ratio)
    labels = np.argmax(close_open_ratio, axis=1)
    num_training_sample = int(T * training_data_ratio)
    return (close_open_ratio[:num_training_sample], labels[:num_training_sample]), \
           (close_open_ratio[num_training_sample:], labels[num_training_sample:])

def train_optimal_action_given_future_obs(model, target_history, target_stocks,
                                          weight_path='weights/optimal_3_stocks.h5'):
    (X_train, y_train), (X_test, y_test) = create_optimal_imitation_dataset(target_history)
    nb_classes = len(target_stocks) + 1

    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    continue_train = True
    while continue_train:
        model.fit(X_train, Y_train, batch_size=128, epochs=20, validation_data=(X_test, Y_test), shuffle=True)
        save_weights = input('Type True to save weights\n')
        if save_weights:
            model.save(weight_path)
        continue_train = input('True to continue train, otherwise stop\n')


def create_network_give_past(nb_classes, window_length, weight_path='weights/imitation_3_stocks.h5'):
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(1, 3), input_shape=(nb_classes, window_length, 1),
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(filters=32, kernel_size=(1, window_length - 2), input_shape=(nb_classes, window_length - 2, 1),
                     activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten(input_shape=(window_length, nb_classes)))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(lr=1e-3),
                  metrics=['accuracy'])
    try:
        model.load_weights(weight_path)
        print('Model load successfully')
    except:
        print('Build model from scratch')
    return model


def train_optimal_action_given_history_obs(model, target_history, target_stocks, window_length,
                                           weight_path='weights/imitation_3_stocks.h5'):
    nb_classes = len(target_stocks) + 1
    (X_train, y_train), (X_validation, y_validation) = create_imitation_dataset(target_history, window_length)
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_validation = np_utils.to_categorical(y_validation, nb_classes)
    X_train = np.expand_dims(X_train, axis=-1)
    X_validation = np.expand_dims(X_validation, axis=-1)
    continue_train = True
    while continue_train:
        model.fit(X_train, Y_train, batch_size=128, epochs=100, validation_data=(X_validation, Y_validation),
                  shuffle=True)
        save_weights = input('Type True to save weights\n')
        if save_weights:
            model.save(weight_path)
        continue_train = input("True to continue train, otherwise stop training...\n")