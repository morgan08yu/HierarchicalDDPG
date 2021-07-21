"""

"""
from __future__ import print_function
from pprint import pprint
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import time
import gym
import gym.spaces
import os

os.sys.path.append(os.path.abspath('.'))
os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system/'))
from utils.data import date_to_index, index_to_date
from utils.notebook_plot import LivePlotNotebook

logger = logging.getLogger(__name__)
eps = 1e-8


def random_shift(x, fraction):
    """ Apply a random shift to a pandas series. """
    min_x, max_x = np.min(x), np.max(x)
    m = np.random.uniform(-fraction, fraction, size=x.shape) + 1
    return np.clip(x * m, min_x, max_x)


def scale_to_start(x):
    """ Scale pandas series so that it starts at one. """
    x = (x + eps) / (x[0] + eps)
    return x


def sharpe(returns, freq=50, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
    return (trough - peak) / (peak + eps)


# step = 2000

class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, df, steps, scale=True, scale_extra_cols=True, augment=0.00, window_length=50, random_reset=True):
        """
        DataSrc.

        df - csv for data frame index of timestamps
             and multi-index columns levels=['open','low','high','close',...]]
             an example is included as an hdf file in this repository
        steps - total steps in episode
        scale - scale the data for each episode
        scale_extra_cols - scale extra columns by global mean and std
        augment - fraction to augment the data by
        random_reset - reset to a random time (otherwise continue through time)

        """
        self.steps = steps + 1
        self.augment = augment
        self.random_reset = random_reset
        self.scale = scale
        self.scale_extra_cols = scale_extra_cols
        self.window_length = window_length
        self.idx = self.window_length
        # dataframe to matrix
        self.asset_names = df.columns.levels[0].tolist()
        self.features = df.columns.levels[1].tolist()
        data = df.values.reshape((len(df), len(self.asset_names), len(self.features)))  # data = (time, asset_names, features)
        self._data = np.transpose(data, (1, 0, 2))  # _data =(asset_names, time, features)
        self._times = df.index

        self.price_columns = ['Close', 'High', 'Low', 'Open']
        self.non_price_columns = set(df.columns.levels[1]) - set(self.price_columns)

        " Normalize non price columns"
        if scale_extra_cols:
            x = self._data.reshape((-1, len(self.features)))
            self.stats = dict(mean=x.mean(0), std=x.std(0))
        self.reset()

    def _step(self):
        # get history matrix from dataframe
        self.step += 1
        data_window = self.data[:, self.step:self.step + self.window_length, :].copy()
        #cprice = self.data[:, self.step:self.step + self.window_length, :].copy()# TODO
        # truth_obs = self._data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        cprice = self.data[:, self.step:self.step + self.window_length, 3]
        "price relative change for closing pricing "
        y1 = data_window[:, -1, 3] / data_window[:, -2, 3]  # features = (open, close, high, low, vol)
        y1 = np.concatenate([[1.0], y1])  # add cash price
        # y1 should be the ('cash', 'stock1', 'stock2','stock3',..)
        # (eq 18) X: prices are divided by close price
        nb_pc = len(self.price_columns)
        if self.scale:
            last_close_price = data_window[:, -1, 3]
            data_window[:, :, :nb_pc] /= last_close_price[:, np.newaxis, np.newaxis]

        if self.scale_extra_cols:
            "normalize non price columns"
            data_window[:, :, nb_pc:] -= self.stats["mean"][None, None, nb_pc:]
            data_window[:, :, nb_pc:] /= self.stats["std"][None, None, nb_pc:]
            data_window[:, :, nb_pc:] = np.clip(data_window[:, :, nb_pc:],
                                                self.stats["mean"][nb_pc:] - self.stats["std"][nb_pc:] * 2,
                                                self.stats["mean"][nb_pc:] + self.stats["std"][nb_pc:] * 2)
        #self.step += 1
        history = (data_window-1)*100
        done = bool(self.step >= self.steps)
        return history, y1, done, cprice

    def reset(self):
        self.step = 0
        "extract data for this episode"
        self.idx = np.random.randint(low=self.window_length + 1, high=self._data.shape[1] - self.steps - 2)
        self.data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :].copy()
        self.times = self._times[self.idx - self.window_length:self.idx + self.steps + 1]



class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    """
    def __init__(self, steps, asset_names=[], trading_cost=0.0025, time_cost=0.0000, utility='MV', gamma=0.01):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        self.utility = utility
        self.gamma = gamma
        self.reset()

    def assets_historical_returns_and_covariances(self, prices):
        #prices = matrix(prices)  # create numpy matrix from prices
        # create matrix of historical returns
        #prices = prices.T
        #prices = np.matrix(prices)
        rows, cols = prices.shape
        # calculate returns
        expreturns = np.array([])
        returns = np.empty([rows, cols - 1])
        for r in range(rows):
            for c in range(cols - 1):
                p0, p1 = prices[r, c], prices[r, c + 1]
                returns[r, c] = (p1 / p0) - 1
        for r in range(rows):
            expreturns = np.append(expreturns, np.mean(returns[r]))
        # calculate covariances
        covars = np.cov(returns)
        expreturns = expreturns*250   # Annualize returns
        covars = covars*250  # Annualize covariances
        return expreturns, covars


    def _step(self, w1, y1, cprice):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        """
        w0 = self.w0
        p0 = self.p0
        d50 = cprice
        d5 =  cprice[:, -5:]
        d10 = cprice[:, -10:]
        d15 = cprice[:, -15:]
        d20 = cprice[:, -20:]
        d30 = cprice[:, -30:]
        d40 = cprice[:, -40:]
        ret5, cor5  =  self.assets_historical_returns_and_covariances(d5)
        ret10, cor10 = self.assets_historical_returns_and_covariances(d10)
        ret15, cor15 = self.assets_historical_returns_and_covariances(d15)
        ret20, cor20 = self.assets_historical_returns_and_covariances(d20)
        ret30, cor30 = self.assets_historical_returns_and_covariances(d30)
        ret40, cor40 = self.assets_historical_returns_and_covariances(d40)
        ret50, cor50 = self.assets_historical_returns_and_covariances(d50)

        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)
        # (eq16) cost to change portfolio
        # (excluding change in cash to avoid double counting for transaction cost)
        c1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum()
        p1 = p0 * (1 - c1) * np.dot(y1, w0)  # final portfolio value
        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding
        # can't have negative holdings in this model (no shorts)
        p1 = np.clip(p1, 0, np.inf)
        rho1 = p1 / p0 - 1  # rate of returns
        # r1 = np.log(p1/p0)  #  log rate of return
        if self.utility == 'Log':  # log rate of return
            r1 = np.log(p1 / p0)
            # reward = r1 * 10000 / self.steps
        elif self.utility == 'Power':  # CRRA power utility function
            r1 = ((p1 / p0) ** self.gamma - 1) / self.gamma
            # reward = r1 * 10000 / self.steps
        elif self.utility == 'Exp':  # exponential utility function
            r1 = (1 - np.exp(-self.gamma * p1 / p0)) / self.gamma
            # reward = r1 * 10000 / self.steps
        elif self.utility == 'MV':
            r_5  = np.dot(w1[1:], ret5) - self.gamma * np.dot(w1[1:].T, np.dot(cor5, w1[1:]))
            r_10 = np.dot(w1[1:], ret10) - self.gamma * np.dot(w1[1:].T, np.dot(cor10, w1[1:]))
            r_15 = np.dot(w1[1:], ret15) - self.gamma * np.dot(w1[1:].T, np.dot(cor15, w1[1:]))
            r_20 = np.dot(w1[1:], ret20) - self.gamma * np.dot(w1[1:].T, np.dot(cor20, w1[1:]))
            r_30 = np.dot(w1[1:], ret30) - self.gamma * np.dot(w1[1:].T, np.dot(cor30, w1[1:]))
            r_40 = np.dot(w1[1:], ret40) - self.gamma * np.dot(w1[1:].T, np.dot(cor40, w1[1:]))
            r_50 = np.dot(w1[1:], ret50) - self.gamma * np.dot(w1[1:].T, np.dot(cor50, w1[1:]))
            #r_min = np.min([r_5, r_10, r_15, r_20, r_30, r_40, r_50])
            #r_max = np.max([r_5, r_10, r_15, r_20, r_30, r_40, r_50])
            r_set = [r_5, r_10, r_15, r_20, r_30, r_40, r_50]
        else:
            raise Exception('Invalid value for utility: %s' % self.utility)
        # immediate reward is log rate of return scaled by episode length
        #reward = r1 * 10000 / self.steps
        reward_set = r_set * 100  #/ self.steps * 100 TODO
        #reward_best = r_max / self.steps * 100
        #reward = [reward_min, reward_max]
        # remember for next step
        self.w0 = w1
        self.p0 = p1
        # if we run out of money, we're done
        done = bool(p1 == 0)
        # should only return single values, not list
        info = {
            #"Reward_set": reward_set,
            #"Reward_max": reward_best,
            "return": rho1,
            "portfolio_value": p1,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": c1,
        }
        # record weights and prices
        for i, name in enumerate(['Cash'] + self.asset_names):
            info['weight_' + name] = w1[i]
            info['price_' + name] = y1[i]
        self.infos.append(info)
        return reward_set, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1] + [0] * len(self.asset_names))
        self.p0 = 1


class RobPortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    """
    metadata = {'render.modes': ['notebook', 'humman']}

    def __init__(self,
                 df,
                 steps,
                 trading_cost=0.0025,
                 time_cost=0.00,
                 window_length=50,
                 augment=0.00,
                 output_mode='EIIE',
                 log_dir=None,
                 scale=True,
                 scale_extra_cols=True,
                 random_reset=True):
        """
        An environment for financial portfolio management.
        Params:
            df - csv for data frame index of timestamps
                 and multi-index columns levels=[['LTCBTC'],...],['open','low','high','close']]
            steps - steps in episode
            window_length - how many past observations["history"] to return
            trading_cost - cost of trade as a fraction,  e.g. 0.0025 corresponding to max rate of 0.25% at Poloniex (2017)
            time_cost - cost of holding as a fraction
            augment - fraction to randomly shift data by
            output_mode: decides observation["history"] shape
            - 'EIIE' for (assets, window, 3)
            - 'atari' for (window, window, 3) (assets is padded)
            - 'mlp' for (assets*window*3)
            log_dir: directory to save plots to
            scale - scales price data by last opening price on each episode (except return)
            scale_extra_cols - scales non price data using mean and std for whole dataset
        """
        self.src = DataSrc(df=df, steps=steps, scale=scale, scale_extra_cols=scale_extra_cols,
                           augment=augment, window_length=window_length,
                           random_reset=random_reset)
        # self._plot = self._plot2 = self._plot3 = None
        self.output_mode = output_mode
        self.sim = PortfolioSim(asset_names=self.src.asset_names,
                                trading_cost=trading_cost, time_cost=time_cost,
                                steps=steps, gamma=1, utility='MV')
        self.log_dir = log_dir
        self._plot = self._plot2 = self._plot3 = None

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(nb_assets + 1,), dtype=np.float32)

        # get the history space from the data min and max
        if output_mode == 'EIIE':
            obs_shape = (nb_assets, window_length, len(self.src.features))
        elif output_mode == 'mlp':
            obs_shape = (nb_assets) * window_length * (len(self.src.features))
        else:
            raise Exception('Invalid value for output_mode: %s' % self.output_mode)

        self.observation_space = gym.spaces.Dict({
            'history': gym.spaces.Box(-np.inf, np.inf, obs_shape),
            'weights': self.action_space})
        self.reset()

    def step(self, action):
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight between 0 and 1. The first (w0) is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(action.shape, (len(self.sim.asset_names) + 1,))
        # action = (action + 1.0)/2.0
        action_take = np.clip(action, 0, 1)
        logger.debug('action: %s', action_take)
        weights = action_take
        weights = weights / (weights.sum() + eps)
        weights[0] += np.clip(1 - weights.sum(), 0, 1)
        # Sanity checks
        assert self.action_space.contains(weights), 'action should be within %r but is %r' % (
        self.action_space, weights)
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3,
                                       err_msg='weights should sum to 1. action="%s"' % weights)

        history, y1, done1, cprice = self.src._step()
        reward_set, info, done2 = self.sim._step(weights, y1, cprice)
        #if self.market_mode == 'Bear':
        #    reward = reward[1]
        #else:
        #    reward = reward[0]

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.src.times[self.src.step].timestamp()
        info['steps'] = self.src.step
        self.infos.append(info)
        # reshape history according to output mode
        if self.output_mode == 'EIIE':
            pass
        elif self.output_mode == 'atari':
            padding = history.shape[1] - history.shape[0]
            history = np.pad(history, [[0, padding], [0, 0], [0, 0]], mode='constant')
        elif self.output_mode == 'mlp':
            history = history.flatten()
        return {'history': history, 'weights': weights}, reward_set, done1 or done2, info

    def reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward_set, done, info = self.step(action)
        return observation

    def render(self, mode='notebook', close=False):
        if mode == 'humman':
            self.plot()
        elif mode == 'notebook':
            self.plot_notebook(close=close)

    def plot_notebook(self, close):
        """Live plot using the jupyter notebook rendering of matplotlib."""
        if close:
            self._plot = self._plot2 = self._plot3 = None
            return
        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')

        # plot prices and performance
        all_assets = ['Cash'] + self.sim.asset_names
        if not self._plot:
            colors = [None] * len(all_assets) + ['black']
            self._plot_dir = os.path.join(self.log_dir,
                                          'notebook_plot_prices_' + str(time.time())) if self.log_dir else None
            self._plot = LivePlotNotebook(log_dir=self._plot_dir, title='prices & performance',
                                          labels=all_assets + ["Portfolio"], ylabel='value', colors=colors)
        x = df_info.index
        y_portfolio = df_info["portfolio_value"]
        y_assets = [df_info['price_' + name].cumprod() for name in all_assets]
        self._plot.update(x, y_assets + [y_portfolio])
        plt.show()

        # plot portfolio weights
        if not self._plot2:
            self._plot_dir2 = os.path.join(self.log_dir,
                                           'notebook_plot_weights_' + str(time.time())) if self.log_dir else None
            self._plot2 = LivePlotNotebook(log_dir=self._plot_dir2, labels=all_assets, title='weights', ylabel='weight')
        ys = [df_info['weight_' + name] for name in all_assets]
        self._plot2.update(x, ys)
        plt.show()

        # plot portfolio costs
        if not self._plot3:
            self._plot_dir3 = os.path.join(self.log_dir,
                                           'notebook_plot_cost_' + str(time.time())) if self.log_dir else None
            self._plot3 = LivePlotNotebook(log_dir=self._plot_dir3, labels=['cost'], title='costs', ylabel='cost')
        ys = [df_info['cost'].cumsum()]
        self._plot3.update(x, ys)
        plt.show()

        if close:
            self._plot = self._plot2 = self._plot3 = None

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value"]].plot(title=title, fig=plt.gcf(), rot=30)
        plt.show()


if __name__ == "__main__":
    def assets_historical_returns_and_covariances(prices):
        #prices = matrix(prices)  # create numpy matrix from prices
        # create matrix of historical returns
        #prices = prices.T
        #prices = np.matrix(prices)
        rows, cols = prices.shape
        returns = np.empty([rows, cols-1])
        for r in range(rows):
            for c in range(cols-1):
                p0, p1 = prices[r, c], prices[r, c+1]
                returns[r, c] = (p1 / p0) - 1
        # calculate returns
        expreturns = np.array([])
        for r in range(rows):
            expreturns = np.append(expreturns, np.mean(returns[r]))
        # calculate covariances
        covars = np.cov(returns)
        expreturns = expreturns * 250  #TODO Annualize returns
        covars = covars*250  # Annualize covariances
        return expreturns, covars


    def ewma_vectorized_2d(data, alpha, axis=None, offset=None, dtype=None, order='C', out=None):
        """
        Calculates the exponential moving average over a given axis.
        :param data: Input data, must be 1D or 2D array.
        :param alpha: scalar float in range (0,1)
            The alpha parameter for the moving average.
        :param axis: The axis to apply the moving average on.
            If axis==None, the data is flattened.
        :param offset: optional
            The offset for the moving average. Must be scalar or a
            vector with one element for each row of data. If set to None,
            defaults to the first value of each row.
        :param dtype: optional
            Data type used for calculations. Defaults to float64 unless
            data.dtype is float32, then it will use float32.
        :param order: {'C', 'F', 'A'}, optional
            Order to use when flattening the data. Ignored if axis is not None.
        :param out: ndarray, or None, optional
            A location into which the result is stored. If provided, it must have
            the same shape as the desired output. If not provided or `None`,
            a freshly-allocated array is returned.
        """
        data = np.array(data, copy=False)

        assert data.ndim <= 2

        if dtype is None:
            if data.dtype == np.float32:
                dtype = np.float32
            else:
                dtype = np.float64
        else:
            dtype = np.dtype(dtype)

        if out is None:
            out = np.empty_like(data, dtype=dtype)
        else:
            assert out.shape == data.shape
            assert out.dtype == dtype

        if data.size < 1:
            # empty input, return empty array
            return out

        if axis is None or data.ndim < 2:
            # use 1D version
            if isinstance(offset, np.ndarray):
                offset = offset[0]
            return ewma_vectorized(data, alpha, offset, dtype=dtype, order=order,
                                   out=out)

        assert -data.ndim <= axis < data.ndim

        # create reshaped data views
        out_view = out
        if axis < 0:
            axis = data.ndim - int(axis)

        if axis == 0:
            # transpose data views so columns are treated as rows
            data = data.T
            out_view = out_view.T

        if offset is None:
            # use the first element of each row as the offset
            offset = np.copy(data[:, 0])
        elif np.size(offset) == 1:
            offset = np.reshape(offset, (1,))

        alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

        # calculate the moving average
        row_size = data.shape[1]
        row_n = data.shape[0]
        scaling_factors = np.power(1. - alpha, np.arange(row_n+1, dtype=dtype),
                                   dtype=dtype)
        # create a scaled cumulative sum array
        np.multiply(data,
            np.multiply(alpha * scaling_factors[-2], np.ones((row_size, 1), dtype=dtype),
                        dtype=dtype) / scaling_factors[np.newaxis, :-1],
            dtype=dtype, out=out_view)
        np.cumsum(out_view, axis=1, dtype=dtype, out=out_view)
        out_view /= scaling_factors[np.newaxis, -2::-1]

        if not (np.size(offset) == 1 and offset == 0):
            offset = offset.astype(dtype, copy=False)
            # add the offsets to the scaled cumulative sums
            out_view += offset[:, np.newaxis] * scaling_factors[np.newaxis, 1:]

        return out


    def ewma_vectorized(data, alpha, offset=None, dtype=None, order='C', out=None):
        """
        Calculates the exponential moving average over a vector.
        Will fail for large inputs.
        :param data: Input data
        :param alpha: scalar float in range (0,1)
            The alpha parameter for the moving average.
        :param offset: optional
            The offset for the moving average, scalar. Defaults to data[0].
        :param dtype: optional
            Data type used for calculations. Defaults to float64 unless
            data.dtype is float32, then it will use float32.
        :param order: {'C', 'F', 'A'}, optional
            Order to use when flattening the data. Defaults to 'C'.
        :param out: ndarray, or None, optional
            A location into which the result is stored. If provided, it must have
            the same shape as the input. If not provided or `None`,
            a freshly-allocated array is returned.
        """
        data = np.array(data, copy=False)

        if dtype is None:
            if data.dtype == np.float32:
                dtype = np.float32
            else:
                dtype = np.float64
        else:
            dtype = np.dtype(dtype)

        if data.ndim > 1:
            # flatten input
            data = data.reshape(-1, order)

        if out is None:
            out = np.empty_like(data, dtype=dtype)
        else:
            assert out.shape == data.shape
            assert out.dtype == dtype

        if data.size < 1:
            # empty input, return empty array
            return out

        if offset is None:
            offset = data[0]

        alpha = np.array(alpha, copy=False).astype(dtype, copy=False)

        # scaling_factors -> 0 as len(data) gets large
        # this leads to divide-by-zeros below
        scaling_factors = np.power(1. - alpha, np.arange(data.size + 1, dtype=dtype),
                                   dtype=dtype)
        # create cumulative sum array
        np.multiply(data, (alpha * scaling_factors[-2]) / scaling_factors[:-1],
                    dtype=dtype, out=out)
        np.cumsum(out, dtype=dtype, out=out)

        # cumsums / scaling
        out /= scaling_factors[-2::-1]

        if offset != 0:
            offset = np.array(offset, copy=False).astype(dtype, copy=False)
            # add offsets
            out += offset * scaling_factors[1:]

        return out
