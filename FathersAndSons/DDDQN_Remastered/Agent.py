import keras.backend as K
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from Memory import ReplayBuffer
from Network import DuelingDeepQNetwork
from tensorflow.python.keras.optimizer_v2.adam import Adam


class Agent:
    def __init__(
        self,
        learning_rate,
        gamma,
        n_actions,
        epsilon,
        batch_size,
        input_dims,
        eps_dec,
        eps_end=0.01,
        mem_size=1000000,
        fname="C:\Research\Dissertation\FathersAndSons\Models",
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
        self.eps_end = eps_end
        self.eps_dec = eps_dec
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
            actions = self.q_eval.A(self.q_eval.dense2(self.q_eval.dense1(state)))
            action = tf.math.argmax(actions, axis=1).numpy()[0]

        return action

    def learn(self, eps_condition):
        if self.memory.mem_counter < self.batch_size:
            return

        if self.learn_step_counter % self.replace == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

        states, actions, rewards, states_, dones = self.memory.sample_buffer(
            self.batch_size
        )

        q_pred = self.q_eval(states)
        q_next = self.q_next(states_)

        q_target = q_pred.numpy()
        max_actions = tf.math.argmax(self.q_eval(states_), axis=1)

        for idx, terminal in enumerate(dones):
            q_target[idx, int(actions[idx])] = rewards[idx] + self.gamma * q_next[
                idx, max_actions[idx]
            ] * (1 - int(dones[idx]))

        self.q_eval.train_on_batch(states, q_target)

        # dynamic epsilon learning instead of static dec. e^-(x-x0) where x0 is the min win.
        # I feel like epsilon jumps too much. At one point it was 1 then 0.9 and then back to 1.
        self.epsilon_decrease(eps_condition)

        self.learn_step_counter += 1

    def save_model(self, name) -> None:
        self.q_eval.save(self.fname + r"\eval\q_eval" + name)
        self.q_next.save(self.fname + r"\next\q_next" + name)
        print("___models saved successfully___")

    def save_model_latest(self, name) -> None:
        self.q_eval.save(self.fname + r"\latest_e\q_eval" + name)
        self.q_next.save(self.fname + r"\latest_n\q_next" + name)
        print("___saved latest models___")

    def load_model(self, name) -> None:
        self.q_eval = keras.models.load_model(self.fname + r"\eval\q_eval" + name)
        self.q_next = keras.models.load_model(self.fname + r"\next\q_next" + name)
        print("___models loaded successfully___")

    def load_model_latest(self, name) -> None:
        self.q_eval = keras.models.load_model(self.fname + r"\latest_e\q_eval" + name)
        self.q_next = keras.models.load_model(self.fname + r"\latest_n\q_next" + name)
        print("___loaded latest models___")

    def epsilon_decrease(self, condition):
        # interim_epsilon = np.exp(-(condition - 0.005))
        # self.epsilon = abs(interim_epsilon) * self.epsilon
        # if self.epsilon > 1:
        #     self.epsilon = 1

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

        # Initialise several NNs with 0 epsilon and then run a series of them. Extract the buy and sell weights, average them and epsilon is the variance of your buy and sell.
