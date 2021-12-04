#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K
import numpy as np
from network import *
import pickle

class OUActionNoise:
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        # Formula taken from https://www.wikipedia.org/wiki/Ornstein-Uhlenbeck_process.
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) *
            np.random.normal(size=self.mean.shape)
        )
        # Store x into x_prev
        # Makes next noise dependent on current one
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class GaussNoise():
    def __init__(self, mean, std_deviation, **kwargs):
        self.mean = mean
        self.std_dev = std_deviation

    def __call__(self):
        return np.random.normal(self.mean, self.std_dev)

    def reset(self):
        pass


class Buffer:
    def __init__(self, num_actions, num_states, buffer_capacity = 100000, batch_size = 64, model_dir = None):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        self.model_dir = model_dir
        # Num of tuples to train on.
        self.batch_size = batch_size

        self.num_actions = num_actions
        self.num_states = num_states

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # We compute the loss and update parameters
    def sample_buffer(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)

    def save_buffer(self, episode):
        np.savez(os.path.join(self.model_dir, f'memory_buffer{episode}'),
                 arr_1=self.state_buffer,
                 arr_2=self.action_buffer,
                 arr_3=self.reward_buffer,
                 arr_4=self.next_state_buffer)
        return

    def load_buffer(self, id):
        npzfile = np.load(os.path.join(self.model_dir, f'memory_buffer{id}.npz'))
        self.state_buffer = npzfile['arr_1']
        self.action_buffer = npzfile['arr_2']
        self.reward_buffer = npzfile['arr_3']
        self.next_state_buffer = npzfile['arr_4']
        temp = np.where(self.reward_buffer == 0)
        if len(temp[0]) == 0:
            self.buffer_counter = self.buffer_capacity
        else:
            self.buffer_counter = temp[0][0]
        return


class Agent():
    def __init__(self, num_actions, num_states, lower_bound, upper_bound,
     batch_size = 32,
     learning_rate_actor=0.001, 
     learning_rate_critic=0.001,
     noise_std_dev=0.3,
     noise_theta=0.15,
     noise_dt=1e-2,
     gamma = 0.99,   
     tau = 0.005,
     model_dir = None,
     training = True):

        self.num_actions = num_actions
        self.num_states = num_states
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Networks
        self.actor = ActorNetwork(num_states, num_actions, upper_bound)
        self.critic = CriticNetwork(num_states, num_actions)
        self.target_actor = ActorNetwork(num_states, num_actions, upper_bound)
        self.target_critic = CriticNetwork(num_states, num_actions)
        # Optimizers
        self.actor_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_actor)
        self.critic_optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate_critic)

        # Parameters
        self.batch_size = batch_size
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.noise_std_dev = noise_std_dev
        self.noise_theta = noise_theta
        self.noise_dt = noise_dt
        self.gamma = gamma
        self.tau = tau
        self.training = training

        self.model_dir = model_dir
        if not self.training:
            try:
                os.makedirs(self.model_dir)
                print("Directory: ", self.model_dir, ". Created")
            except FileExistsError:
                print("Directory: ", self.model_dir, ". Already exists")

            self.buffer = Buffer(num_actions, num_states, batch_size=batch_size, model_dir = model_dir)
            
        self.noise_object = OUActionNoise(mean=np.zeros(num_actions), std_deviation=noise_std_dev, theta=noise_theta, dt=noise_dt)

    def sample_action(self, state):
        sampled_actions = tf.squeeze(self.actor.model(state, training=False))
        noise = self.noise_object()
        # Adding noise to action
        #print(noise)
        if self.training:
            sampled_actions = sampled_actions.numpy() + noise
        else:
            sampled_actions = sampled_actions.numpy()

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, self.lower_bound, self.upper_bound)

        return [np.squeeze(legal_action)]

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(self, state_batch, action_batch, reward_batch, next_state_batch):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        # TRAIN CRITIC NETWORK
        with tf.GradientTape() as tape:
            target_actions = self.target_actor.model(next_state_batch, training=False)
            y = reward_batch + self.gamma * self.target_critic.model(
                [next_state_batch, target_actions], training=True
            )
            critic_value = self.critic.model(
                [state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(
            critic_loss, self.critic.model.trainable_variables)
        self.critic_optimizer.apply_gradients(
            zip(critic_grad, self.critic.model.trainable_variables)
        )

        # TRAIN ACTOR NETWORK
        with tf.GradientTape() as tape:
            actions = self.actor.model(state_batch, training=True)
            critic_value = self.critic.model([state_batch, actions], training=False)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, self.actor.model.trainable_variables)
        self.actor_optimizer.apply_gradients(
            zip(actor_grad, self.actor.model.trainable_variables)
        )

        # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer.buffer_counter,
                           self.buffer.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.buffer.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(
            self.buffer.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(
            self.buffer.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(
            self.buffer.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


    @tf.function
    def update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(b * self.tau + a * (1 - self.tau))

    def save_checkpoint(self, id, overwrite=False):
        # Save models
        self.actor.model.save_weights(
            self.model_dir + f'/actor{id}.h5', overwrite=overwrite)
        self.critic.model.save_weights(
            self.model_dir + f'/critic{id}.h5', overwrite=overwrite)
        self.target_actor.model.save_weights(
            self.model_dir + f'/target_actor{id}.h5', overwrite=overwrite)
        self.target_critic.model.save_weights(
            self.model_dir + f'/target_critic{id}.h5', overwrite=overwrite)

        # Save optimizers
        symbolic_weights = getattr(self.actor_optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(self.model_dir + f'/actor_optimizer{id}.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

        symbolic_weights = getattr(self.critic_optimizer, 'weights')
        weight_values = K.batch_get_value(symbolic_weights)
        with open(self.model_dir + f'/critic_optimizer{id}.pkl', 'wb') as f:
            pickle.dump(weight_values, f)

    def load_checkpoint(self, file = None, id = ""):
        # Load models
        if file is None:
            file = self.model_dir

        grad_vars_act = self.actor.model.trainable_weights
        grad_vars_crit = self.critic.model.trainable_weights
        zero_grads_act = [tf.zeros_like(w) for w in grad_vars_act]
        zero_grads_crit = [tf.zeros_like(w) for w in grad_vars_crit]

        with open(file+ f'/actor_optimizer{id}.pkl', 'rb') as f:
            actor_weight_values = pickle.load(f)
        with open(file + f'/critic_optimizer{id}.pkl', 'rb') as f:
            critic_weight_values = pickle.load(f)

        # Apply gradients which don't do nothing with Adam
        self.actor_optimizer.apply_gradients(
            zip(zero_grads_act, grad_vars_act))
        self.critic_optimizer.apply_gradients(
            zip(zero_grads_crit, grad_vars_crit))

        self.actor_optimizer.set_weights(actor_weight_values)
        self.critic_optimizer.set_weights(critic_weight_values)

        self.actor.model.load_weights(file + f'/actor{id}.h5')
        self.critic.model.load_weights(file + f'/critic{id}.h5')
        self.target_actor.model.load_weights(file + f'/target_actor{id}.h5')
        self.target_critic.model.load_weights(file + f'/target_critic{id}.h5')
