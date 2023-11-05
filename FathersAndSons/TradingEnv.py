import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding
import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.models import load_model
from utils import plotLearning
from numba import jit, cuda
import time


# Keeps track of states, actions, rewards and samples them at random.
class ReplayBuffer(object):
    def __init__(self, max_size, input_size):
        self.mem_size = max_size
        self.mem_counter = (
            0  # This counter is the running index of the last stored memory.
        )
        self.state_memory = np.zeros((self.mem_size, *input_size))
        self.new_state_memory = np.zeros((self.mem_size, *input_size))
        self.action_memory = np.zeros(self.mem_size)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)

    def store_transition(self, state, action, reward, state_, done):
        index = (
            self.mem_counter % self.mem_size
        )  # this will loop around as we overwrite the end of the buffer. So, the buffer will have 1 million possible entries. Once we reach a million it'll go back around to position 0.
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[
            index
        ] = done  # True evaluates to 1. So we do the opposite.
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        max_mem = min(
            self.mem_counter, self.mem_size
        )  # we want to know the position of the lasdt stored memory so that we only sample up to that position
        batch = np.random.choice(max_mem, batch_size, replace=False)
        # will sample integers between max_mem --> batch_size. Replace=False means if you sample a memory then you can't sample it again

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, states_, terminal


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


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        super(DuelingDeepQNetwork, self).__init__()
        self.dense1 = keras.layers.Dense(fc1_dims, activation="relu")
        self.dense2 = keras.layers.Dense(fc2_dims, activation="relu")
        self.V = keras.layers.Dense(1, activation=None)  # value of state
        self.A = keras.layers.Dense(
            n_actions, activation=None
        )  # advantage layer - adv of each action given state and action
        # gpus = tf.config.experimental.list_physical_devices('GPU')
        # tf.config.experimental.set_memory_growth(gpus, True)

    def call(self, state):  # feedforward from pytorch (What?)
        x = self.dense1(state)
        x = self.dense2(x)
        V = self.V(x)
        A = self.A(x)

        Q = V + (A - tf.math.reduce_mean(A, axis=1, keepdims=True))

        return Q

    def advantage(self, state):
        x = self.dense1(state)
        x = self.dense2(x)
        A = self.A(x)

        return A


class Agent:
    def __init__(
        self,
        learning_rate,
        gamma,
        n_actions,
        epsilon,
        batch_size,
        input_dims,
        epsilon_dec=1e-3,
        eps_end=0.01,
        mem_size=1000,
        fname="dueling_dqn.h5",
        fc1_dims=128,
        fc2_dims=128,
        replace=100,
    ):
        # The replace parameter is a hyperparameter that comes into play when discussing the online evaluation network and the target network.
        # In DQN we are using one policy - eps greedy - to generate sample data for updating our model of the world.
        # But our model of the world is updated based on the estimate for the purely greedy value. The maximum set of actions for a given state.
        # A DQN is a temporal difference learning method as it learns at every step. So if you're using a target value to update something generated else (there's hugging face page to explain this better). You're chasing a moving target, that results in model instability.
        # The replace parameter - every 100 moves we replace our weights of the target network with the weights of our online evaluation network.
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_end = eps_end
        self.fname = fname
        self.replace = replace
        self.batch_size = batch_size
        self.learn_step_counter = (
            0  # tells us that it's time to update the parameters for our update network
        )
        self.memory = ReplayBuffer(mem_size, input_dims)
        self.q_eval = DuelingDeepQNetwork(fc1_dims, fc2_dims, n_actions)
        self.q_next = DuelingDeepQNetwork(fc1_dims, fc2_dims, n_actions)

        self.q_eval.compile(
            optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error"
        )

        self.q_next.compile(
            optimizer=Adam(learning_rate=learning_rate), loss="mean_squared_error"
        )

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.advantage(state)
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    # @jit(target_backend="cuda")
    def learn(self):
        if self.memory.mem_counter < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size
        )

        q_pred = self.q_eval(states)
        q_next = tf.math.reduce_max(self.q_next(states_), axis=1, keepdims=True).numpy()
        q_target = np.copy(q_pred)

        for idx, terminal in enumerate(dones):
            if terminal:
                q_next[idx] = 0.0
            q_target[idx, int(actions[idx])] = rewards[idx] + self.gamma * q_next[idx]
        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end
        )

        self.learn_step_counter += 1

    def save_model(self):
        self.q_eval.save_weights(self.fname)
        self.q_next.save_weights(self.fname)
        # self.q_eval.save(self.fname)

    def load_model(self):
        self.q_eval.load_weights(self.fname)
        self.q_next.load_weights(self.fname)
        # self.q_eval = load_model(self.fname)


def run(filename, train=True, n_games=1):
    env = TradeEnv(trading_cost=0, train=train)

    agent = Agent(
        gamma=0.999,
        epsilon=1.0,
        learning_rate=1e-3,
        input_dims=[6],
        epsilon_dec=1e-3,
        mem_size=1000,
        batch_size=64,
        eps_end=0.01,
        fc1_dims=128,
        fc2_dims=128,
        replace=100,
        n_actions=3,
    )

    if not train:
        n_steps = 0
        while n_steps < agent.batch_size:
            observation = env.reset()
            action = env.action_space.sample()
            observation_, reward, done, info = env.step(action, observation)
            agent.store_transition(observation, action, reward, observation_, done)
            n_steps += 1
        agent.learn()
        agent.load_model()

    scores, eps_history = [], []

    for i in range(n_games):
        done = False
        EpRewards = 0
        observation = env.reset()
        while not done:
            action = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action, observation)
            EpRewards += reward
            agent.store_transition(observation, action, reward, observation_, done)
            observation = observation_
            if train:
                agent.learn()
        eps_history.append(agent.epsilon)
        scores.append(EpRewards)

        avg_score = np.mean(scores)
        print(
            "episode: ",
            i,
            "score %.1f" % EpRewards,
            "average score %.1f" % avg_score,
            "epsilon %.2f" % agent.epsilon,
        )

    if train:
        agent.save_model()
    x = [i + 1 for i in range(n_games)]
    plotLearning(x, scores, eps_history, filename)


if __name__ == "__main__":
    np.random.seed(42)
    run("ThirdTry_100.png", True, n_games=100)
    run("Test.png", False, n_games=100)
