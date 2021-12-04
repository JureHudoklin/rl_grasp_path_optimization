#! /usr/bin/python3
# -*- coding: utf-8 -*-

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class ActorNetwork():
    def __init__(self, num_states, num_actions, upper_bound) -> None:
        self.num_states = num_states
        self.upper_bound = upper_bound
        self.num_actions = num_actions
        self.get_actor()
        
    def get_actor(self):
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(3, self.num_states))
        out = layers.LSTM(64)(inputs)
        out = layers.Dense(128)(out)
        out = layers.BatchNormalization()(out)
        out = layers.Activation('relu')(out)
        out = layers.Dense(32, activation="relu")(out)
        outputs = layers.Dense(self.num_actions, activation="tanh",
                            kernel_initializer=last_init)(out)


        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * self.upper_bound
        model = tf.keras.Model(inputs, outputs)
        self.model = model





class CriticNetwork():
    def __init__(self, num_states, num_actions) -> None:
        self.num_states = num_states
        self.num_actions = num_actions
        self.get_critic()

    def get_critic(self):
        # State as input
        state_input = layers.Input(shape=(3, self.num_states))
        state_out = layers.LSTM(64)(state_input)
        state_out = layers.Dense(32)(state_out)
        state_out = layers.BatchNormalization()(state_out)
        state_out = layers.Activation("relu")(state_out)
        state_out = layers.Dense(64, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(self.num_actions))
        action_out = layers.Dense(64)(action_input)
        action_out = layers.BatchNormalization()(action_out)
        action_out = layers.Activation("relu")(action_out)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])
        out = layers.Dense(128, activation="relu")(concat)
        out = layers.Dense(32, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        self.model = model
