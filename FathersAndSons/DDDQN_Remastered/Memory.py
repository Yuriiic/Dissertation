import numpy as np


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
