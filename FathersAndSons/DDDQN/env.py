import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.python.keras as keras
import time
from gymnasium import spaces
from gymnasium.utils import seeding
from numba import jit, cuda
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.optimizer_v2.adam import Adam


class Data:
    def __init__(self, timeframe, ticker, train=True):
        self.timeframe = timeframe  # timeframe of the data you wish to view - 1min, 15min, 30min, etc
        self.ticker = ticker  # which data you want to use
        self.data = self.load_data()
        self.preprocess_data(self.timeframe, train)
        self.step = 0
        self.offset = 0

    def load_data(self):
        print(f"Loading {self.ticker} Data")
        if self.ticker == "Oil":
            df = pd.read_csv(
                rf"C:\Research\Dissertation\Datasets\{self.ticker}_Summary.csv.gz"
            )
        elif self.ticker == "Wheat":
            df = pd.read_csv(
                rf"C:\Research\Dissertation\Datasets\{self.ticker}_Summary.csv.gz"
            )
        else:
            raise ValueError("Unrecognised ticker")
        print("---------------------")
        print(f"Finished Loading {self.ticker} Data")
        return df

    def preprocess_data(self, timeframe, train):
        """
        The preprocess function right now just converts the data into the timeframe of interest.
        In the future, you can include technical analysis components such as MACD, RSI, etc.
        Right now though we only have OHLCV and the 1 day returns
        :param timeframe:
        :return:
        """

        self.data.rename(columns={"Date-Time": "Period", "Last": "Close"}, inplace=True)
        # self.data = self.data[['Period', 'Open', 'High', 'Low', 'Last', 'Volume', 'No. Trades', 'Close Bid', 'No. Bids', 'Close Ask', 'No. Asks', 'Close Bid Size', 'Close Ask Size']]
        self.data = self.data[["Period", "Open", "High", "Low", "Close", "Volume"]]
        self.data["Period"] = pd.to_datetime(self.data["Period"])
        self.data.set_index("Period", inplace=True)
        if self.timeframe == "1M":
            pass
        elif self.timeframe == "D":
            self.data = self.data.resample(timeframe)
            agg_funcs = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
            self.data = self.data.agg(agg_funcs)
        else:
            raise ValueError("Currently Unsupported Timeframe")

        self.data.reset_index(inplace=True)
        self.data["Period"] = self.data["Period"].dt.tz_localize(None)
        self.data["Weekend"] = (self.data["Period"].dt.dayofweek >= 5).astype(bool)
        self.data = self.data[self.data["Weekend"] == False]
        self.data = self.data.loc[:, self.data.columns != "Weekend"]
        self.data = self.data.dropna()
        self.data["Period"] = pd.to_datetime(self.data["Period"])
        self.data["Period"] = (
            self.data["Period"].dt.year * 10**8
            + self.data["Period"].dt.month * 10**6
            + self.data["Period"].dt.day * 10**4
            + self.data["Period"].dt.hour * 10**2
            + self.data["Period"].dt.minute
        )
        if train:
            self.data = self.data.head(int(len(self.data) * (0.8)))
        else:
            self.data_train = self.data.head(int(len(self.data) * (0.8)))
            self.data_test = self.data[~self.data.isin(self.data_train)].dropna()
            self.data = self.data_test

    def reset(self):
        # NOTE: check the index that days aren't missing.
        self.data.reset_index(drop=True, inplace=True)

        # I am not sure what this is about
        # I think it's to randomly pluck samples out of the data. This specifically is to start at a random place
        high = len(self.data.index)
        self.offset = np.random.randint(low=0, high=high)
        self.step = 0

    def take_step(self):
        """Returns data for current trading day and done signal"""
        obs = self.data.iloc[self.offset + self.step]
        self.step += 1

        done = self.data.index[-1] == (self.offset + self.step)
        return obs, done


class TradeEnv(gym.Env):
    def __init__(self, trading_cost, train=True):
        self.data_source = Data(timeframe="1M", ticker="Oil", train=train)
        self.trading_cost = trading_cost
        self.action_space = spaces.Discrete(3)

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation):
        assert self.action_space.contains(
            action
        ), f"{action} {type(action)} is an invalid action"

        action -= 1

        pos = observation.Close

        observation_, done = self.data_source.take_step()

        reward = action * (observation_.Close - pos) / pos

        return observation_, reward, done, None

    def reset(self):
        self.data_source.reset()

        return self.data_source.take_step()[0]

    def render(self):
        """Will implement"""
        pass

