#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
from replay_buffer import Agent
from environment import *
import numpy as np
import tensorflow as tf
import datetime
import time


if __name__ == "__main__":
    start_time = time.time()
    # solve tensorflow memory issue
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(
        physical_devices[0], True)  # allow memory growth

    environment = Environment(
        "/home/jure/programming/reinforcement_learning_project/meshes/object_5.obj")

    # ----------------- Where to save/load the models from-----------------
    #model_dir = "/home/jure/programming/reinforcement_learning_project/checkpoints/20211123-235121"
    cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = "/home/jure/programming/reinforcement_learning_project/checkpoints"
    model_dir = os.path.join(model_dir, cur_date)
    #---------------------------------------------------------------------------------
    ddpg_agent = Agent(4, 6, -np.pi, np.pi,
                       batch_size=128,
                       learning_rate_actor=0.0005, learning_rate_critic=0.0005,
                       noise_std_dev=0.3,
                       noise_theta=0.15,
                       noise_dt=0.01,
                       gamma=0.99,
                       tau=0.001,
                       model_dir=model_dir)

    #ddpg_agent.load_checkpoint()
    #ddpg_agent.buffer.load_buffer(950)

    num_of_episodes = 100001
    episode_number = 1 # 151
    ep_reward_list = []
    avg_reward_list = []
    best_score = -np.inf

    while episode_number < num_of_episodes:
        failed_episode = False
        state = environment.reset()
        episodic_reward = 0
        step_i = 1
        while True:
            tf_state = tf.expand_dims(tf.convert_to_tensor(state), 0)
            action = ddpg_agent.sample_action(tf_state)[0]
            new_state, reward, done, info = environment.step(action)
           
            ddpg_agent.buffer.record((state, action, reward, new_state))
            
            ddpg_agent.learn()
            ddpg_agent.update_target(ddpg_agent.target_actor.model.variables,
                          ddpg_agent.actor.model.variables)
            ddpg_agent.update_target(ddpg_agent.target_critic.model.variables,
                          ddpg_agent.critic.model.variables)

            episodic_reward += reward
            state = new_state
            step_i += 1
            if done or step_i > 200:
                break
            

        episodic_reward = episodic_reward / step_i

        if episode_number % 10000 == 0:
            # ddpg_agent.buffer.save_buffer(episode_number)
            # ddpg_agent.save_checkpoint(episode_number)
            np.savez(model_dir + f"/{episode_number}",
                    {"episodic_reward" : np.array(ep_reward_list)})
            #ddpg_agent.noise_object.reset()

        if episode_number % 10 == 0:
            ddpg_agent.noise_object.reset()
            ddpg_agent.noise_object.std_dev -= 0.0002

        print("Number of steps in episode", step_i)

        ep_reward_list.append(episodic_reward)

        # Mean of last 40 episodes
        avg_reward = np.mean(ep_reward_list[-100:])

        if avg_reward > best_score:
            best_score = avg_reward
            ddpg_agent.buffer.save_buffer("best")
            ddpg_agent.save_checkpoint("best", overwrite=True)

        print("Episode * {} * Avg Reward is ==> {}".format(episode_number, avg_reward))
        avg_reward_list.append(avg_reward)

        episode_number += 1

    print("--------------------------------------------------------")
    print("Finished calculations in: %s seconds" % (time.time() - start_time))
    print("-------------------------END----------------------------")
