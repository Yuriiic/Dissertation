import tensorflow as tf
import tensorflow.python.keras as keras
from tensorflow.python.keras.optimizer_v2.adam import Adam


class DuelingDeepQNetwork(keras.Model):
    def __init__(self, fc1_dims, fc2_dims, n_actions):
        # fc1 and 2 are fully connected layers
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
