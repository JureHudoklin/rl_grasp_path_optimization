from buffer import ReplayBuffer
from noise import OUActionNoise
from networks import ActorNetwork, CriticNetwork
import torch.nn.functional as F
import torch as T
import os
from rl_task.srv import RLstep, RLstepResponse
from std_srvs import Empty, EmptyResponse

import rospy
import numpy as np


class SimulationEnvironment(object):
    def __init__(self):

        # Create service client
        self.env_step_srv = rospy.ServiceProxy('/rl_step_srv', RLStep)
        self.env_reset_srv = rospy.ServiceProxy('/rl_reset_srv', Empty)

    def env_step(self, action):
        """
        Performs one step of the environment.
        """
        # Send the action
        step_action = RLstep()
        step_action.action = action
        # Get response
        step_response = self.env_step_srv(step_action)
        # Format the response
        new_state = step_response.state
        reward = step_response.reward
        done = step_response.done
        info = step_response.info
        # Return the result
        return new_state, reward, done, info

    def env_reset(self):
        self.env_reset_srv()
        return None


class DDPG_agent(object):
    def __init__(self):


if __name__ == "__main__":
    rospy.init_node('agent_node')

    ddpg_agent = DDPGAgent()
    while not rospy.is_shutdown():

        rospy.spin()
