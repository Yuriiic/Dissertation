import numpy as np
import tensorflow as tf
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
        epsilon_dec=1e-3,
        eps_end=0.01,
        mem_size=1000,
        fname="dueling_ddqn",
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

    def save_models(self) -> None:
        self.q_eval.save_weights(self.fname + "_q_eval_weights.h5")
        q_eval_symbolic_weights = getattr(
            self.q_eval.optimizer, "Models/dueling_ddqn_q_eval_weights"
        )
        q_eval_weight_values = K.batch_get_value(q_eval_symbolic_weights)
        with open("Models/eval_optimizer.pkl", "wb") as f:
            pickle.dump(q_eval_weight_values, f)

        self.q_next.save_weights(self.fname + "_q_next_weights.h5")
        q_next_symbolic_weights = getattr(
            self.q_next.optimizer, "Models/dueling_ddqn_q_next_weights"
        )
        q_next_weight_values = K.batch_get_value(q_next_symbolic_weights)
        with open("Models/next_optimizer.pkl", "wb") as f:
            pickle.dump(q_next_weight_values, f)

        print("... models saved successfully ...")

    def load_models(self):
        self.q_eval.load_weights(self.fname + "_q_eval_weights.h5")
        self.q_eval._make_train_function()
        with open("Models/eval_optimizer.pkl", "rb") as f:
            eval_weight_values = pickle.load(f)
        self.q_eval.optimizer.set_weights(eval_weight_values)

        self.q_next.load_weights(self.fname + "_q_next_weights.h5")
        self.q_next._make_train_function()
        with open("Models/next_optimizer.pkl", "rb") as f:
            next_weight_values = pickle.load(f)
        self.q_next.optimizer.set_weights(next_weight_values)
        print("... models loaded successfully ...")
