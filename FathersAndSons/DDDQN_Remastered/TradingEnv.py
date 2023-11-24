import gymnasium as gym
import numpy as np
from Data import Data
from Memory import ReplayBuffer
from gymnasium import spaces


class TradeEnv(gym.Env):
    def __init__(self, trading_cost, timeframe="1M", ticker="Oil", train=True):
        self.data_source = Data(timeframe=timeframe, ticker=ticker, train=train)
        self.trading_cost = trading_cost
        self.action_space = spaces.Discrete(2)

    def seed(self, seed=42):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, observation):
        assert self.action_space.contains(
            action
        ), f"{action} {type(action)} is an invalid action"

        if action == 0:
            action = -1

        pos = observation.Close

        observation_, done = self.data_source.take_step()

        reward = action * (observation_.Close - pos)

        return observation_, reward, done, None

    def reset(self):
        self.data_source.reset()

        return self.data_source.take_step()[0]

    def render(self):
        """Will implement"""
        pass
