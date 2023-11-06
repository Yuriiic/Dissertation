import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from network import DuelingDeepQNetwork
from replay_memory import ReplayBuffer
from tensorflow.keras.optimizers import Adam


class Agent:
    def __init__(
        self,
        gamma,
        epsilon,
        lr,
        n_actions,
        input_dims,
        mem_size,
        batch_size,
        eps_min=0.01,
        eps_dec=5e-7,
        replace=1000,
        chkpt_dir="Models/duelingdoubledqn_",
    ):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(n_actions)]
        self.learn_step_counter = 0
        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
        self.fname = self.chkpt_dir

        self.q_eval = DuelingDeepQNetwork(input_dims, n_actions)
        self.q_eval.compile(optimizer=Adam(learning_rate=lr))
        self.q_next = DuelingDeepQNetwork(input_dims, n_actions)
        self.q_next.compile(optimizer=Adam(learning_rate=lr))

    def save_models(self) -> None:
        self.q_eval.save_weights(self.fname + "q_eval_weights.h5")
        q_eval_symbolic_weights = getattr(
            self.q_eval.optimizer, "Models/duelingdoubledqn_q_eval_weights"
        )
        q_eval_weight_values = K.batch_get_value(q_eval_symbolic_weights)
        with open("Models/eval_optimizer.pkl", "wb") as f:
            pickle.dump(q_eval_weight_values, f)

        self.q_next.save_weights(self.fname + "q_next_weights.h5")
        q_next_symbolic_weights = getattr(
            self.q_next.optimizer, "Models/duelingdoubledqn_q_next_weights"
        )
        q_next_weight_values = K.batch_get_value(q_next_symbolic_weights)
        with open("Models/next_optimizer.pkl", "wb") as f:
            pickle.dump(q_next_weight_values, f)

        print("... models saved successfully ...")

    def load_models(self):
        self.q_eval.load_weights(self.fname + "q_eval_weights.h5")
        self.q_eval._make_train_function()
        with open("Models/eval_optimizer.pkl", "rb") as f:
            eval_weight_values = pickle.load(f)
        self.q_eval.optimizer.set_weights(eval_weight_values)

        self.q_next.load_weights(self.fname + "q_next_weights.h5")
        self.q_next._make_train_function()
        with open("Models/next_optimizer.pkl", "rb") as f:
            next_weight_values = pickle.load(f)
        self.q_next.optimizer.set_weights(next_weight_values)
        print("... models loaded successfully ...")

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = tf.convert_to_tensor([observation])
            _, advantage = self.q_eval(state)
            action = tf.math.argmax(advantage, axis=1).numpy()[0]
        else:
            action = np.random.choice(self.action_space)
        return action

    def store_transition(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def sample_memory(self):
        state, action, reward, new_state, done = self.memory.sample_buffer(
            self.batch_size
        )
        states = tf.convert_to_tensor(state)
        rewards = tf.convert_to_tensor(reward)
        dones = tf.convert_to_tensor(done)
        actions = tf.convert_to_tensor(action, dtype=tf.int32)
        states_ = tf.convert_to_tensor(new_state)
        return states, actions, rewards, states_, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.set_weights(self.q_eval.get_weights())

    def decrement_epsilon(self):
        self.epsilon = (
            self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min
        )

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        self.replace_target_network()

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = tf.range(self.batch_size, dtype=tf.int32)
        action_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            V_s, A_s = self.q_eval(states)
            V_s_, A_s_ = self.q_next(states_)
            V_s_eval, A_s_eval = self.q_eval(states_)

            advantage = V_s + A_s - tf.reduce_mean(A_s, axis=1, keepdims=True)
            advantage_ = V_s_ + A_s_ - tf.reduce_mean(A_s_, axis=1, keepdims=True)
            advantage_eval = (
                V_s_eval + A_s_eval - tf.reduce_mean(A_s_eval, axis=1, keepdims=True)
            )
            max_actions = tf.argmax(advantage_eval, axis=1, output_type=tf.int32)
            max_action_idx = tf.stack([indices, max_actions], axis=1)
            q_next = tf.gather_nd(advantage_, indices=max_action_idx)
            q_pred = tf.gather_nd(advantage, indices=action_indices)

            q_target = rewards + self.gamma * q_next * (1 - dones.numpy())
            loss = keras.losses.MSE(q_pred, q_target)
        params = self.q_eval.trainable_variables
        grads = tape.gradient(loss, params)
        self.q_eval.optimizer.apply_gradients(zip(grads, params))
        self.learn_step_counter += 1

        self.decrement_epsilon()
