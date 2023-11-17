import gymnasium as gym
import numpy as np
from Data import Data
from Memory import ReplayBuffer
from gymnasium import spaces


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

        reward = action * (observation_.Close - pos)

        return observation_, reward, done, None

    def reset(self):
        self.data_source.reset()

        return self.data_source.take_step()[0]

    def render(self):
        """Will implement"""
        pass
