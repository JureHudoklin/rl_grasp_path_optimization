#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
from replay_buffer_lstm import Agent
import numpy as np
import tensorflow as tf
import datetime
import time
import rospy

from rl_task.srv import RLStep, RLStepResponse
from rl_task.srv import RLReset, RLResetResponse


class SimulationEnvironment(object):
    def __init__(self):

        # Create service client
        self.env_step_srv = rospy.ServiceProxy('/rl_step_srv', RLStep)
        self.env_reset_srv = rospy.ServiceProxy('/rl_reset_srv', RLReset)
        rospy.sleep(1)

    def step(self, action):
        """
        Performs one step of the environment.
        """
        # Send the action
        step_action = [action[0], action[1], action[2], action[3], 0]
        # Get response
        step_response = self.env_step_srv(step_action)
        # Format the response
        new_state = np.array(step_response.state)
        reward = np.array(step_response.reward)
        done = np.array(step_response.done)
        info = np.array(step_response.info)
        # Return the result
        return new_state, reward, done, info

    def reset(self):
        response = self.env_reset_srv([0.1, 0.1, 0.1, 0.1, 0.1])
        start_state = np.array(response.state)
        return start_state


if __name__ == "__main__":
    start_time = time.time()
    rospy.init_node('agent_node')
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True)  # allow memory growth

    environment = SimulationEnvironment()

    # ----------------- Where to save/load the models from-----------------
    model_dir = "/home/jure/reinforcement_ws/src/ddpg_rl_agent/src/checkpoints_lstm/20211203-224505"
    #cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #model_dir = "/home/jure/programming/reinforcement_learning_project/checkpoints"
    #model_dir = os.path.join(model_dir, cur_date)
    #---------------------------------------------------------------------------------
    ddpg_agent = Agent(4, 6, -1, 1,
                       batch_size=64,
                       learning_rate_actor=0.001, learning_rate_critic=0.001,
                       noise_std_dev=0.1,
                       noise_theta=0.15,
                       noise_dt=0.01,
                       gamma=0.99,
                       tau=0.001,
                       model_dir=model_dir)

    ddpg_agent.load_checkpoint(id="best")
    ddpg_agent.buffer.load_buffer("best")

    num_of_episodes = 100
    episode_number = 0  # 151
    ep_reward_list = []
    avg_reward_list = []
    states = []
    best_score = -np.inf

    while episode_number < num_of_episodes:
        failed_episode = False

        # Reset the environment
        state = environment.reset()
        states.append(state)
        old_state = state
        old_state_tf = tf.expand_dims(tf.convert_to_tensor(state), 0)
        episodic_reward = 0
        step_i = 0
        while True:

            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = ddpg_agent.sample_action(old_state_tf, tf_state, tr=False)[0]

            new_state, reward, done, info = environment.step(action)
            print(new_state)

            if info == "fail":
                failed_episode = True
                break

            episodic_reward += reward
            state = new_state
            old_state = state
            old_state_tf = tf.expand_dims(tf.convert_to_tensor(state), 0)
            states.append(state)
            if done:
                break

            step_i += 1

        if failed_episode or episode_number < 0:
            episode_number += 1
            continue

        if episode_number % 1 == 0:
            test_1_dir = "/home/jure/reinforcement_ws/src/ddpg_rl_agent/src"
            np.save(test_1_dir + f"/coopelia_test{episode_number}.npz",
                    np.array(states))
            states = []


        print("Number of steps in episode", step_i)

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-40:])
        print("Episode * {} * Avg Reward is ==> {}".format(episode_number, avg_reward))
        avg_reward_list.append(avg_reward)

        episode_number += 1

    print("--------------------------------------------------------")
    print("Finished calculations in: %s seconds" % (time.time() - start_time))
    print("-------------------------END----------------------------")

 

