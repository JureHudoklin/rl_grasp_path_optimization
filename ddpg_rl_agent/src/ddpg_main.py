#! /usr/bin/python3
# -*- coding: utf-8 -*-

import os
from ddpg_agent import Agent
from rl_task.srv import RLStep, RLStepResponse
from rl_task.srv import RLReset, RLResetResponse

import rospy
import numpy as np
import datetime


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
        step_action = [action[0]/5, action[1]/5, action[2]/5, 0, 0]
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

    # ----------------- Where to save/load the models from-----------------
    #model_dir = "/home/jure/reinforcement_ws/src/ddpg_rl_agent/src/checkpoints/20211120-180856"
    cur_date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    model_dir = "/home/jure/reinforcement_ws/src/ddpg_rl_agent/src/checkpoints/20211120-230404"
    model_dir = os.path.join(model_dir, cur_date)
    #---------------------------------------------------------------------------------
    ddpg_agent = Agent(0.01, 0.01, 6, 0.001, 3, batch_size=32,model_dir=model_dir)
    environment = SimulationEnvironment()
    #environment.env_reset()

    num_of_episodes = 1000
    episode_number = 100
    score_history = []
    best_score = -np.inf

    #ddpg_agent.load_models()


    while episode_number < num_of_episodes:
        # Reset the environment
        state = environment.env_reset()
        done = False
        score = 0
        step_i = 0
        while not done:
            step_i += 1
            action = ddpg_agent.choose_action(state)
            new_state, reward, done, info = environment.env_step(action)
            print(reward)
            if info == "fail":
                break
            if reward < -10:
                break
            ddpg_agent.remember(state, action, reward, new_state, done)
            ddpg_agent.learn()
            score += reward
            state = new_state
        score = score/step_i
        score_history.append(score)
        avg_score = np.mean(score_history[-20:])

        if avg_score > best_score:
            best_score = avg_score
            ddpg_agent.save_models()
        
        if episode_number % 10 == 0:
            print(score_history, " SCORE HISTORY")
            print('Episode: %d, Average Score: %f, Best Score: %f' % (episode_number, avg_score, best_score))
            np.save(model_dir + f"/{episode_number}.npz", np.array(score_history))
        

        print('episode ', episode_number, 'score %.5f' %
              score, 'average score %.5f' % avg_score)
        episode_number += 1
