"""

"""

from __future__ import print_function
from pprint import pprint
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
from scipy.stats import norm
import time
import gym
import gym.spaces
import os

#os.sys.path.append(os.path.abspath('.'))
#os.sys.path.append(os.path.abspath('/Users/Morgans/Desktop/trading_system/'))
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


def sharpe(returns, freq=252, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.values.argmax():].min()
    return (trough - peak) / (peak + eps)


def CVaR(Returns, VaR):
    Time = len(Returns)
    First_Windows = Time - len(VaR)
    CVaR = pd.Series(index=Returns.index, name='CVaR')
    for i in range(0, Time - First_Windows):
        VaR_Return = pd.concat([Returns[First_Windows:], VaR], axis=1)
        Expected_Shortfall = \
        (VaR_Return[:i + 1].ix[VaR_Return[:i + 1].T.iloc[0] < VaR_Return[:i + 1].T.iloc[1]]).T.iloc[0].mean()
        CVaR[First_Windows + i] = Expected_Shortfall
    return pd.DataFrame(CVaR[First_Windows:])


# step = 2000

class DataSrc(object):
    """Acts as data provider for each new episode."""

    def __init__(self, df, steps, scale=True, scale_extra_cols=False, augment=0.00,
                 window_length=50, random_reset=True):
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
        data = df.values.reshape(
            (len(df), len(self.asset_names), len(self.features)))  # data = (time, asset_names, features)
        self._data = np.transpose(data, (1, 0, 2))  # _data =(asset_names, time, features)
        self._times = df.index

        self.price_columns = ['Open', 'High', 'Low', 'Close']
        self.non_price_columns = set(df.columns.levels[1]) - set(self.price_columns)

        " Normalize non price columns"
        if scale_extra_cols:
            x = self._data.reshape((-1, len(self.features)))
            # x = self._data[:, :, 4]
            self.stats = dict(mean=x.mean(0), std=x.std(0))
        self.reset()

    def _step(self):
        # get history matrix from dataframe
        self.step += 1
        data_window = self.data[:, self.step:self.step + self.window_length, :].copy()
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
            #data_window[:, :, :nb_pc] = (data_window[:, :, :nb_pc] - 1)*100

        if self.scale_extra_cols:
            "normalize non price columns"
            data_window[:, :, nb_pc:] -= self.stats["mean"][None, None, nb_pc:]
            data_window[:, :, nb_pc:] /= self.stats["std"][None, None, nb_pc:]
            data_window[:, :, nb_pc:] = np.clip(data_window[:, :, nb_pc:],
                                                self.stats["mean"][nb_pc:] - self.stats["std"][nb_pc:] * 10,
                                                self.stats["mean"][nb_pc:] + self.stats["std"][nb_pc:] * 10)
            # data_window[:, :, nb_pc:] = (data_window[:, :, nb_pc:] - 1) * 100
        history = data_window  # -1) * 100
        done = bool(self.step >= self.steps)
        return history, y1, done, cprice

    def reset(self):
        self.step = 0
        "extract data for this episode"
        if self.random_reset:
            self.idx = np.random.randint(low=self.window_length + 1, high=self._data.shape[1] - self.steps - 2) # TODO modify the low and high
        else:
            # if self.idx > (self._data.shape[1] - self.steps - self.window_length - 1):
            self.idx = self.window_length + 1
            # else:
                # self.idx += self.steps
        # self.idx = np.random.randint(low=self.window_length + 1, high=self._data.shape[1] - self.steps - 2)
        self.data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :].copy()
        self.times = self._times[self.idx - self.window_length:self.idx + self.steps + 1]


class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    """

    def __init__(self, steps, asset_names=[], trading_cost=0.0025, time_cost=0.0000, utility='Log', gamma=1.5, c = 0.04):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        self.utility = utility
        self.gamma = gamma
        self.alpha = 0.05
        self.c = c
        self.reset()

    def assets_historical_returns_and_covariances(self, prices):
        # prices = matrix(prices)  #  create numpy matrix from prices
        # create matrix of historical returns
        # prices = prices.T
        # prices = np.matrix(prices)
        rows, cols = prices.shape
        returns = np.empty([rows, cols - 1])
        for r in range(rows):
            for c in range(cols - 1):
                p0, p1 = prices[r, c], prices[r, c + 1]
                returns[r, c] = (p1 / p0) - 1  # TODO
        # calculate returns
        expreturns = np.array([])
        for r in range(rows):
            expreturns = np.append(expreturns, np.mean(returns[r]))
        # calculate covariances
        covars = np.cov(returns)
        expreturns_anu = (1 + expreturns) ** 250 - 1  # Annualize returns
        covars_anu = covars * 250  # Annualize covariances
        return expreturns, covars

    def ewma_vectorized(self, data, alpha, offset=None, dtype=None, order='C', out=None):
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

    def ewma(self, data, alpha):
        rows, cols = data.shape
        ret = np.empty([4, 50])
        for r in range(rows):
            out = self.ewma_vectorized(data[r, :], alpha=alpha)
            out = np.array([out])
            ret[r, :] = out
        return ret

    def _step(self, w1,  w2, y1, cprice):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        """
        w0 = self.w0
        p0 = self.p0
        p0_gt = self.p0_gt  # portfolio for Dagent
        # p0_at = self.p0_at  # portfolio for Hagent
        w0_gt = self.w0_gt
        # w0_at = self.w0_at

        d50 = cprice
        ret50, cor50 = self.assets_historical_returns_and_covariances(d50)
        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)
        dw1_gt = (y1 * w0_gt) / (np.dot(y1, w0_gt) + eps)  # under ddpg
        # (excluding change in cash to avoid double counting for transaction cost)
        # w_pre = np.array([1] + [0] * len(self.asset_names))
        # dw_pre = (y1 * w_pre) / (np.dot(y1, w_pre) + eps)
        c1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum() # cost for action1
        c1_gt = self.cost * (np.abs(dw1_gt[1:] - w1[1:])).sum() # cost for ddpg
        c2 = self.cost * (np.abs(dw1[1:] - w2[1:])).sum() # cost for action2
        # c_pre = self.cost * (np.abs(dw_pre[1:] - w_pre[1:])).sum()
        p1 = p0 * (1 - c1) * np.dot(y1, w0)
        p2 = p0 * (1 - c2) * np.dot(y1, w0)
        # p1_null = p0 * (1 - c_pre) * np.dot(y1, w0)  # final portfolio value
        p1 = p1 * (1 - self.time_cost) # portfolio value for action1
        p2 = p2 * (1 - self.time_cost) # portfolio value for action2
        p1_gt = p0_gt * (1 - c1_gt) * np.dot(y1, w0_gt)  # DDPG portfolio value
        # can't have negative holdings in this model (no shorts)
        p1 = np.clip(p1, 0, np.inf) # define the portfolio value under action 1
        p2 = np.clip(p2, 0, np.inf) # define the portfolio value under action 2
        p1_gt = np.clip(p1_gt, 0, np.inf)


        rho1 = p1 / p0 - 1  # rate of returns under action1
        rho2 = p2 / p0 - 1 # rate of returns under action2
        rho_DDPG = p1_gt / p0_gt - 1 # rate of returns under ddpg
        mu_gt = np.dot(w1[1:], ret50) # \mu comes from w1 (g_t)
        sigma_gt = np.sqrt(np.dot(w1[1:].T, np.dot(cor50, w1[1:]))) #  sigma comes form w1 (g_t)
        mu_at = np.dot(w2[1:], ret50)  # mu comes from w2 (a_t)
        sigma_at = np.sqrt(np.dot(w2[1:].T, np.dot(cor50, w2[1:])))  # sigma comes form w2(a_t)

        CVaR_n = self.alpha ** -1 * norm.pdf(norm.ppf(self.alpha)) * sigma_gt - mu_gt
        # CVaR is depend on g_t  --- under ddpg
        # r1 = np.log(p1/p0)  #  log rate of return
        CVaR_at = self.alpha ** -1 * norm.pdf(norm.ppf(self.alpha)) * sigma_at - mu_at
        # CVaR is depend on a_t
        sharp_ratio_DDPG = np.sqrt(252) * mu_gt / (sigma_gt + eps)
        sharp_ratio_at = np.sqrt(252) * mu_at / (sigma_at + eps)
        if self.utility == 'Log':  # log rate of return
            r1 = np.log(p1 / p0) # - np.log(y1.mean())  #np.log
            r1_gt = np.log(p1_gt / p0_gt) # - np.log(y1.mean())
            r2 = np.log(p2 / p0) # - np.log(y1.mean())
            reward_gt = r1_gt * 1000 / self.steps
            # reward_at = 1/self.gamma * ( mu_at /(sigma_at+1e-12) -mu_gt /(sigma_gt+1e-12)) #/self.steps
            reward_at = (CVaR_n - CVaR_at) * 1000 / self.steps  # TODO reward for cvar
            # reward_at = (sharp_ratio_at/(c2+eps) - sharp_ratio_DDPG/(c1_gt+eps)) / self.steps
        elif self.utility == 'Power':  # CRRA power utility function
            r1 = ((p1 / p0) ** self.gamma - 1) / self.gamma
            reward = r1 * 10000 / self.steps
        elif self.utility == 'Exp':  # exponential utility function
            r1 = (1 - np.exp(-self.gamma * p1 / p0)) / self.gamma
            reward = r1 * 10000 / self.steps
        elif self.utility == 'MV':
            # r_50 = np.dot(w1[1:], ret50) - self.gamma * np.dot(w1[1:].T, np.dot(cor50, w1[1:]))
            r_50 = np.dot(w1[1:], ret50) - CVaR_n
            r1 = np.log(p1 / p0)
            reward = r_50 * 100 / self.steps  # * 100 / self.steps
        else:
            raise Exception('Invalid value for utility: %s' % self.utility)
        # immediate reward is log rate of return scaled by episode length
        # reward = r_50 * 10000 / self.steps
        # remember for next step
        if CVaR_n < self.c:  # TODO < if using CVaR_n
            self.w0 = w1
            self.p0 = p1
            portfolio_value = p1
            cost = c1
            action_final = w1
            sharp_ratio_final = sharp_ratio_DDPG
            CVaR_final = CVaR_n
            rho = rho1
            z = 0 # z =0 cvar<c
        else:
            self.w0 = w2
            self.p0 = p2
            portfolio_value = p2
            cost = c2
            sharp_ratio_final = sharp_ratio_at
            CVaR_final = CVaR_at
            action_final = w2
            rho = rho2
            z = 1
        # if we run out of money, we're done
        done = bool(p1 == 0)
        #self.p0_at = p2
        self.p0_gt = p1_gt
        self.w0_gt = w1
        # should only return single values, not list
        info = {
            "reward_gt": reward_gt,
            "reward_at": reward_at,
            "log-return_gt": r1,
            "log-return_at": r2,
            "portfolio_value": portfolio_value,
            "portfolio_value_DDPG": p1_gt,
            # "portfolio_value_at": p2,
            "market_return": y1.mean(),
            "rate_of_return": rho,
            "rate_of_return DDPG": rho_DDPG,
            # "weights_gt_mean": w1.mean(),
            # "weights_gt_std": w1.std(),
            # "weights_at_mean": w2.mean(),
            # "weights_at_std": w2.std(),
            "cost": cost,
            "cost_DDPG": c1_gt,
            "CVaR_DDPG": CVaR_n,
            "CVaR_H": CVaR_at,
            "state": z,
            "Sharp ratio DDPG": sharp_ratio_DDPG,
            "Sharp ratio": sharp_ratio_final,
            "CVaR": CVaR_final
        }
        # record weights and prices
        for i, name in enumerate(['Cash'] + self.asset_names):
            info['weight_gt_' + name] = w1[i]
            info['weight_at_' + name] = w2[i]  # info['price_gt_' + name] = y1[i]
        self.infos.append(info)
        for i, name in enumerate(['Cash'] + self.asset_names):
            info['weight_' + name] = action_final[i]
            info['price_' + name] = y1[i]
        self.infos.append(info)
        return reward_gt, reward_at, info, done, z

    def reset(self):
        self.infos = []
        self.w0 = np.array([1] + [0] * len(self.asset_names))
        self.p0 = self.p0_gt = self.p0_at = 1
        self.w0_gt = np.array([1] + [0] * len(self.asset_names))
        self.w0_at = np.array([1] + [0] * len(self.asset_names))


class PPortfolioEnv(gym.Env):
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
                 utility='Log',
                 gamma=2,
                 c = 0.045,
                 output_mode='EIIE',
                 log_dir = None,
                 scale = True,
                 scale_extra_cols = False,
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
                           augment=augment, window_length=window_length, random_reset=random_reset)
        self.output_mode = output_mode
        self.sim = PortfolioSim(asset_names=self.src.asset_names, utility = utility, gamma = gamma, c = c,
                                trading_cost=trading_cost, time_cost=time_cost, steps=steps)
        self.log_dir = log_dir
        self._plot = self._plot2 = self._plot3 = self._plot4 = self._plot5 = None

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(nb_assets + 1,))  # , dtype = np.float32)
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

    def step(self, action1, action2):
        # TODO action1 = g_t (lower-level hierachical/actor of DDPG)
        # TODO action2 = a_t (high-level hierachical network)
        # TODO z state the Cvar, z=1 cvar > c, z=0 cvar<c
        """
        Step the env.
        Actions should be portfolio [w0...]
        - Where wn is a portfolio weight between 0 and 1. The first (w0) is cash_bias
        - cn is the portfolio conversion weights see PortioSim._step for description
        """
        np.testing.assert_almost_equal(action1.shape, (len(self.sim.asset_names) + 1,))
        # action = (action + 1.0)/2.0
        action_take1 = np.clip(action1, 0, 1)
        logger.debug('action: %s', action_take1)
        np.testing.assert_almost_equal(action2.shape, (len(self.sim.asset_names) + 1,))
        action_take2 = np.clip(action2, 0, 1)
        logger.debug('action: %s', action_take2)
        weights1 = action_take1
        weights1 = weights1 / (weights1.sum() + eps)
        weights1[0] += np.clip(1 - weights1.sum(), 0, 1)
        weights2 = action_take2
        weights2 = weights2 / (weights2.sum() + eps)
        weights2[0] += np.clip(1 - weights2.sum(), 0, 1)
        # Sanity checks
        assert self.action_space.contains(weights1), 'action should be within %r but is %r' % (self.action_space, weights1)
        np.testing.assert_almost_equal(np.sum(weights1), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights1)
        assert self.action_space.contains(weights2), 'action should be within %r but is %r' % (self.action_space, weights2)
        np.testing.assert_almost_equal(np.sum(weights2), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights2)

        history, y1, done1, cprice = self.src._step()
        reward_gt, reward_at, info, done2, z = self.sim._step(weights1, weights2, y1, cprice)
        # TODO output include z and two reward function

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
        return {'history': history, 'weights1': weights1}, reward_gt, reward_at, done1 or done2, info, z
# TODO remove weights2
    def reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward_gt, reward_at, done, info, z = self.step(action, action)
        return observation

    def render(self, mode='notebook', close=False):
        if mode == 'humman':
            self.plot()
        elif mode == 'notebook':
            self.plot_notebook(close=close)

    def plot_notebook(self, close):
        """Live plot using the jupyter notebook rendering of matplotlib."""
        if close:
            self._plot = self._plot2 = self._plot3 = self._plot4 = self._plot5 = None
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
            self._plot_dir3 = os.path.join(self.log_dir,'notebook_plot_cost_' + str(time.time())) if self.log_dir else None
            self._plot3 = LivePlotNotebook(log_dir=self._plot_dir3, labels=['cost']+['cost_DDPG'], title='costs', ylabel='cost')
        ys = [df_info['cost'].cumsum()]
        cost = [df_info['cost_DDPG'].cumsum()]
        self._plot3.update(x, ys + cost)
        plt.show()

        if not self._plot4:
            self._plot_dir4 = os.path.join(self.log_dir, 'notebook_plot_sharpratio_' + str(time.time())) if self.log_dir else None
            self._plot4 = LivePlotNotebook(log_dir=self._plot_dir4, labels=['Sharp ratio']+['Sharp ratio DDPG'], title='Sharp ratio', ylabel='Sharp ratio')
        ys = [df_info['Sharp ratio']]
        yss = [df_info['Sharp ratio DDPG']]
        self._plot4.update(x, ys + yss)
        plt.show()

        if not self._plot5:
            self._plot_dir5 = os.path.join(self.log_dir, 'notebook_plot_CVaR_' + str(time.time())) if self.log_dir else None
            self._plot5 = LivePlotNotebook(log_dir=self._plot_dir5, labels=['CVaR']+['CVaR_DDPG'], title='CVaR', ylabel= 'CVaR')
        ys = [df_info['CVaR']]
        CVaR_DDPG = [df_info['CVaR_DDPG']]
        self._plot5.update(x, ys + CVaR_DDPG)
        plt.show()

        if close:
            self._plot = self._plot2 = self._plot3 = self._plot4 = self._plot5 = None

    def plot(self):
        # show a plot of portfolio vs mean market performance
        df_info = pd.DataFrame(self.infos)
        df_info.index = pd.to_datetime(df_info["date"], unit='s')
        rate_return = df_info["portfolio_value"].iloc[-1]-1
        mdd = max_drawdown(df_info.rate_of_return + 1)
        sharpe_ratio = sharpe(df_info.rate_of_return)
        title = 'rate_return={:2.2%} max_drawdown={: 2.2%} sharpe_ratio={: 2.4f}'.format(rate_return, mdd, sharpe_ratio)
        df_info[["portfolio_value", "market_value", "portfolio_value_DDPG"]].plot(title=title, fig=plt.gcf(), rot=30)
        plt.show()





#if __name__ == "__main__":
# df_train = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/four_stocks/poloniex_vol_4.hf', key='train')
# df_test = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/four_stocks/poloniex_vol_4.hf', key='test')

# df = DataSrc(df_train, 256, scale=True, scale_extra_cols=True, augment=0.00, window_length=50, random_reset=True)


