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


def sharpe(returns, freq=50, rfr=0):
    """ Given a set of returns, calculates naive (rfr=0) sharpe (eq 28). """
    return (np.sqrt(freq) * np.mean(returns - rfr + eps)) / np.std(returns - rfr + eps)


def max_drawdown(returns):
    """ Max drawdown. See https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp """
    peak = returns.max()
    trough = returns[returns.argmax():].min()
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
                 window_length=50, random_reset=True, include_cash=True):
        """
        DataSrc.
        df: (num_stock, steps, features), which include open, high, low, close and volumn
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
        self.include_cash = include_cash
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
        data_window = self.data[:, self.step:self.step + self.window_length, :].copy()  # TODO
        ground_truth_obs = self.data[:, self.step + self.window_length:self.step + self.window_length + 1, :].copy()
        # TODO adding the next step as the truth obs to train cnn to find the initial point for DDPG.
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
            close_price_truth_obs = ground_truth_obs[:, -1, 3]
            ground_truth_obs[:, :, :nb_pc] /= close_price_truth_obs[:, np.newaxis, np.newaxis]
        if self.scale_extra_cols:
            "normalize non price columns"
            last_volumn = data_window[:, -1, 4]
            data_window[:, :, nb_pc:] /= last_volumn[:, np.newaxis, np.newaxis]
            last_volumn_truth_obs = ground_truth_obs[:, -1, 4]
            ground_truth_obs[:, :, nb_pc:] /= last_volumn_truth_obs[:, np.newaxis, np.newaxis]
            #data_window[:, :, nb_pc:] -= self.stats["mean"][None, None, nb_pc:]  # TODO
            #data_window[:, :, nb_pc:] /= self.stats["std"][None, None, nb_pc:]  # TODO
            #data_window[:, :, nb_pc:] = np.clip(data_window[:, :, nb_pc:],
                                                #self.stats["mean"][nb_pc:] - self.stats["std"][nb_pc:] * 2,
                                                #self.stats["mean"][nb_pc:] + self.stats["std"][nb_pc:] * 2)
            #data_window[:, :, nb_pc:] = (data_window[:, :, nb_pc:] - 1) * 100
        history = data_window  # -1) * 100
        if self.include_cash:
            if self.scale and self.scale_extra_cols:
                cash_his = np.ones((1, self.window_length, history.shape[2]))
                cash_obs = np.ones((1, 1, ground_truth_obs.shape[2]))
            if not self.scale and not self.scale_extra_cols:
                cash_his = np.ones((1, self.window_length, history.shape[2]))
                cash_obs = np.ones((1, 1, ground_truth_obs.shape[2]))
            history = np.concatenate((cash_his, history), axis=0)
            ground_truth_obs = np.concatenate((cash_obs, ground_truth_obs), axis=0)
        done = bool(self.step >= self.steps)
        return history, y1, done, cprice, ground_truth_obs

    def reset(self):
        self.step = 0
        "extract data for this episode"
        if self.random_reset:
            self.idx = np.random.randint(low=self.window_length + 1, high=self._data.shape[1] - self.steps - 2) # TODO modify the low and high
        else:
            self.idx = self.window_length + 1
        self.data = self._data[:, self.idx - self.window_length:self.idx + self.steps + 1, :].copy()
        self.times = self._times[self.idx - self.window_length:self.idx + self.steps + 1]
        return self.data[:, self.step:self.step + self.window_length,:].copy(),\
               self.data[:, self.step+self.window_length:self.step + self.step + 1, :]

class PortfolioSim(object):
    """
    Portfolio management sim.
    Params:
    - cost e.g. 0.0025 is max in Poliniex
    """
    def __init__(self, steps,  asset_names=[], trading_cost=0.0025, time_cost=0.0000, utility='Log', gamma=1):
        self.cost = trading_cost
        self.time_cost = time_cost
        self.steps = steps
        self.asset_names = asset_names
        self.utility = utility
        self.gamma = gamma
        self.alpha = 0.05
        self.reset()

    def assets_historical_returns_and_covariances(self, prices):
        #prices = matrix(prices)  # create numpy matrix from prices
        # create matrix of historical returns
        #prices = prices.T
        #prices = np.matrix(prices)
        rows, cols = prices.shape
        returns = np.empty([rows, cols-1])
        for r in range(rows):
            for c in range(cols-1):
                p0, p1 = prices[r, c], prices[r, c+1]
                returns[r, c] = (p1 / p0) - 1  # TODO
        # calculate returns
        expreturns = np.array([])
        for r in range(rows):
            expreturns = np.append(expreturns, np.mean(returns[r]))
        # calculate covariances
        covars = np.cov(returns)
        expreturns_anu = (1+expreturns)**250 -1   # Annualize returns
        covars_anu = covars*250  # Annualize covariances
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
            out = self.ewma_vectorized(data[r,:], alpha=alpha)
            out = np.array([out])
            ret[r,:] = out
        return ret

    def _step(self, w1, y1, cprice):
        """
        Step.
        w1 - new action of portfolio weights - e.g. [0.1,0.9, 0.0]
        y1 - price relative vector also called return
            e.g. [1.0, 0.9, 1.1]
        """
        w0 = self.w0  # TODO here, the w0 represent the old action.
        p0 = self.p0
        d50 = cprice
        ret50, cor50 = self.assets_historical_returns_and_covariances(d50)
        dw1 = (y1 * w0) / (np.dot(y1, w0) + eps)  # TODO check the equtation
        # (excluding change in cash to avoid double counting for transaction cost)
        c1 = self.cost * (np.abs(dw1[1:] - w1[1:])).sum() # TODO check the equation
        p1 = p0 * (1 - c1) * np.dot(y1, w0)
        p1 = p1 * (1 - self.time_cost)  # we can add a cost to holding
        # can't have negative holdings in this model (no shorts)
        p1 = np.clip(p1, 0, np.inf)
        rho1 = p1 / p0 - 1  # rate of returns
        mu_cvar = np.dot(w1[1:], ret50)
        sigma_cvar = np.sqrt(np.dot(w1[1:].T, np.dot(cor50, w1[1:])))
        CVaR_n = self.alpha ** -1 * norm.pdf(norm.ppf(self.alpha)) * sigma_cvar - mu_cvar
        if self.utility == 'Log':
            r1 = np.log(p1 / p0)
            reward = r1 / self.steps
        elif self.utility == 'Power':  # CRRA power utility function
            r1 = ((p1 / p0) ** self.gamma - 1) / self.gamma
            reward = r1 / self.steps
        elif self.utility == 'Exp':   # exponential utility function
            r1 = (1-np.exp(-self.gamma * p1 / p0)) / self.gamma
            reward = r1 / self.steps
        elif self.utility == 'MV':
            # r_50 = np.dot(w1[1:], ret50) - self.gamma * np.dot(w1[1:].T, np.dot(cor50, w1[1:]))
            r_50 = np.dot(w1[1:], ret50) - CVaR_n
            r1 = np.log(p1 / p0)
            reward = r_50 * 100 / self.steps # * 100 / self.steps
        else:
            raise Exception('Invalid value for utility: %s' % self.utility)
        # immediate reward is log rate of return scaled by episode length
        # remember for next step
        self.w0 = w1
        self.p0 = p1

        # if we run out of money, we're done
        done = bool(p1 == 0)
        # should only return single values, not list
        info = {
            "reward": reward,
            "log-return": r1,
            "portfolio_value": p1,
            "market_return": y1.mean(),
            "rate_of_return": rho1,
            "weights_mean": w1.mean(),
            "weights_std": w1.std(),
            "cost": c1,
            "CVaR": CVaR_n
        }
        # record weights and prices
        for i, name in enumerate(['Cash'] + self.asset_names):
            info['weight_' + name] = w1[i]
            info['price_' + name] = y1[i]
        self.infos.append(info)
        return reward, info, done

    def reset(self):
        self.infos = []
        self.w0 = np.array([1] + [0] * len(self.asset_names))
        self.p0 = 1




class PortfolioEnv(gym.Env):
    """
    An environment for financial portfolio management.
    Financial portfolio management is the process of constant redistribution of a fund into different
    financial products.
    Based on [Jiang 2017](https://arxiv.org/abs/1706.10059)
    """
    metadata = {'render.modes': ['notebook', 'humman']}
    def __init__(self, df, steps, trading_cost=0.0025, time_cost=0.00, window_length=50,
                 augment=0.00, utility='Log', gamma=2, output_mode='EIIE', log_dir=None,
                 scale=False, scale_extra_cols=False, random_reset=True, include_cash=True):
        """
        An environment for financial portfolio management.
        Params:
            df - csv for data frame index ,
                             and multi-index columns levels=[['GOOGLE'],["Inter"],...],['open','low','high','close']]
            steps - steps in episode
            window_length - how many past observations["history"] to return
            trading_cost - cost of trade as a fraction,  e.g. 0.0025 corresponding to max rate of 0.25% at Poloniex (2017)
            time_cost - cost of holding as a fraction
            augment - fraction to randomly shift data by
            output_mode: decides observation["history"] shape
            - EIIE for (assets, window, 3)
            - 'atari' for (window, window, 3) (assets is padded)
            - 'mlp' for (assets*window*3)
            log_dir: directory to save plots to
            scale - scales price data by last opening price on each episode (except return)
            scale_extra_cols - scales non price data using mean and std for whole dataset
        """
        self.src = DataSrc(df=df, steps=steps, scale=scale, scale_extra_cols=scale_extra_cols,
                           augment=augment, window_length=window_length, random_reset=random_reset, include_cash = include_cash)
        self.output_mode = output_mode
        self.sim = PortfolioSim(asset_names=self.src.asset_names, utility=utility, gamma=gamma,
                                trading_cost=trading_cost,  time_cost=time_cost, steps=steps)
        self.log_dir = log_dir
        self._plot = self._plot2 = self._plot3 = None

        # openai gym attributes
        # action will be the portfolio weights [cash_bias,w1,w2...] where wn are [0, 1] for each asset
        nb_assets = len(self.src.asset_names)
        self.action_space = gym.spaces.Box(0.0, 1.0, shape=(nb_assets + 1,), dtype=np.float64)
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
        action_take = np.clip(action, 0, 1)
        logger.debug('action: %s', action_take)
        weights = action_take
        weights = weights / (weights.sum())
        weights[0] += np.clip(1 - weights.sum(), 0, 1)
        # Sanity checks
        assert self.action_space.contains(weights), 'action should be within %r but is %r' % (self.action_space, weights)
        np.testing.assert_almost_equal(np.sum(weights), 1.0, 3, err_msg='weights should sum to 1. action="%s"' % weights)

        history, y1, done1, cprice, ground_truth_obs = self.src._step()
        reward, info, done2 = self.sim._step(weights, y1, cprice)

        # calculate return for buy and hold a bit of each asset
        info['market_value'] = np.cumprod([inf["market_return"] for inf in self.infos + [info]])[-1]
        # add dates
        info['date'] = self.src.times[self.src.step].timestamp()
        info['steps'] = self.src.step
        info['next_obs'] = ground_truth_obs # TODO adding the next_obs for training cnn.
        self.infos.append(info)
        # reshape history according to output mode
        if self.output_mode == 'EIIE':
            pass
        elif self.output_mode == 'atari':
            padding = history.shape[1] - history.shape[0]
            history = np.pad(history, [[0, padding], [0, 0], [0, 0]], mode='constant')
        elif self.output_mode == 'mlp':
            history = history.flatten()
        return {'history': history, 'weights': weights}, reward, done1 or done2, info, ground_truth_obs

    def reset(self):
        self.sim.reset()
        self.src.reset()
        self.infos = []
        action = self.sim.w0
        observation, reward, done, info = self.step(action)
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
            self._plot_dir = os.path.join(self.log_dir, 'notebook_plot_prices_' + str(time.time())) if self.log_dir else None
            self._plot = LivePlotNotebook(log_dir=self._plot_dir, title='prices & performance', labels=all_assets + ["Portfolio"], ylabel='value', colors=colors)
        x = df_info.index
        y_portfolio = df_info["portfolio_value"]
        y_assets = [df_info['price_' + name].cumprod() for name in all_assets]
        self._plot.update(x, y_assets + [y_portfolio])
        plt.show()

        # plot portfolio weights
        if not self._plot2:
            self._plot_dir2 = os.path.join(self.log_dir, 'notebook_plot_weights_' + str(time.time())) if self.log_dir else None
            self._plot2 = LivePlotNotebook(log_dir=self._plot_dir2, labels=all_assets, title='weights', ylabel='weight')
        ys = [df_info['weight_' + name] for name in all_assets]
        self._plot2.update(x, ys)
        plt.show()

        # plot portfolio costs
        if not self._plot3:
            self._plot_dir3 = os.path.join(self.log_dir, 'notebook_plot_cost_' + str(time.time())) if self.log_dir else None
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





#if __name__ == "__main__":
# df_train = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/four_stocks/poloniex_vol_4.hf', key='train')
# df_test = pd.read_hdf('/Users/Morgans/Desktop/trading_system/HFT_data/four_stocks/poloniex_vol_4.hf', key='test')
# df = DataSrc(df_train, 256, scale=True, scale_extra_cols=True, augment=0.00, window_length=50, random_reset=True)


