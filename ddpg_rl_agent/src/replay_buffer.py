import os
import numpy as np

class ActionNoise(object):
    def __init__(self, sigma, mu):
        """
        Add noise to the action
        ----------
        Args:
            sigma (float): Standard deviation of the noise
            mu (float): Mean of the noise
        ----------
        """
        self.mu = mu
        self.sigma = sigma

    def __call__(self):
        return self.mu + self.sigma * np.random.randn(1)

class ReplayBuffer(object):
    def __init__(self, buff_mem_size, input_shape, n_actions):
        self.buff_mem_size = buff_mem_size
        self.mem_counter = 0
        self.state_memory = np.zeros((self.buff_mem_size, input_shape))
        self.new_state_memory = np.zeros((self.buff_mem_size, input_shape))
        self.action_memory = np.zeros((self.buff_mem_size, n_actions))
        self.reward_memory = np.zeros(self.buff_mem_size)
        self.terminal_memory = np.zeros(self.buff_mem_size, dtype=np.bool)

    def store_transition(self, state, action, reward, next_state, done):
        # If we input index higher than memory size go to beginning
        index = self.mem_counter % self.buff_mem_size
        # Store state, new state, action, reward and episode done flag
        self.state_memory[index] = state
        self.new_state_memory[index] = next_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - done
        # Update the memory counter
        self.mem_counter += 1

    def sample_buffer(self, batch_size):
        # Check how much data is available
        max_mem = min(self.mem_counter, self.buff_mem_size)
        # Sample a batch of transitions from one state to next
        batch = np.random.choice(max_mem, batch_size, replace=False)
        # Get the batch of data
        states = self.state_memory[batch]
        next_state = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        terminal = self.terminal_memory[batch]

        return states, actions, rewards, next_state, terminal

        
        
