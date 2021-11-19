#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
from ddpg_agent import Agent
from rl_task.srv import RLStep, RLStepResponse
from rl_task.srv import RLReset, RLResetResponse

import rospy
import numpy as np


class SimulationEnvironment(object):
    def __init__(self):

        # Create service client
        self.env_step_srv = rospy.ServiceProxy('/rl_step_srv', RLStep)
        self.env_reset_srv = rospy.ServiceProxy('/rl_reset_srv', RLReset)
        rospy.sleep(1)

    def env_step(self, action):
        """
        Performs one step of the environment.
        """
        # Send the action
        step_action = list(action)
        # Get response
        step_response = self.env_step_srv(step_action)
        # Format the response
        new_state = np.array(step_response.state)
        reward = np.array(step_response.reward)
        done = np.array(step_response.done)
        info = np.array(step_response.info)
        # Return the result
        return new_state, reward, done, info

    def env_reset(self):
        response = self.env_reset_srv([0.1, 0.1, 0.1, 0.1, 0.1])
        start_state = np.array(response.state)
        return start_state


if __name__ == "__main__":
    rospy.init_node('agent_node')

    ddpg_agent = Agent(0.01, 0.01, 6, 0.001, 5)
    environment = SimulationEnvironment()
    #environment.env_reset()

    num_of_episodes = 100
    episode_number = 0
    score_history = []
    best_score = -np.inf


    while episode_number < num_of_episodes:
        # Reset the environment
        state = environment.env_reset()
        done = False
        score = 0
        while not done:
            action = ddpg_agent.choose_action(state)
            new_state, reward, done, info = environment.env_step(action)
            if info == "fail":
                break
            print(new_state, "new_state")
            print(reward, "reward")
            print(done, "done")
            print(info, "info")
            ddpg_agent.remember(state, action, reward, new_state, done)
            ddpg_agent.learn()
            score += reward
            state = new_state
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        if avg_score > best_score:
            best_score = avg_score
            #ddpg_agent.save_models()

        print('episode ', episode_number, 'score %.1f' %
              score, 'average score %.1f' % avg_score)

