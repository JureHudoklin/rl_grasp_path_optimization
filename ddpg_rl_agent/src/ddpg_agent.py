
import os
import numpy as np
import torch as T
import torch.nn.functional as F
from network import ActorNetwork, CriticNetwork
from replay_buffer import ReplayBuffer, ActionNoise
class Agent():
    def __init__(self, alpha, beta, input_dims, tau, n_actions, gamma=0.99,
                 max_size=1000000, fc1_dims=400, fc2_dims=300,
                 batch_size=64):
        self.gamma = gamma # How fast the parameters get copied to target network
        self.tau = tau #??? DIscount rate
        self.batch_size = batch_size # How many experiences to sample from memory
        self.alpha = alpha # Learning rate for actor
        self.beta = beta # Learning rate for critic

        self.memory = ReplayBuffer(max_size, input_dims, n_actions)

        self.noise = ActionNoise(0.05, 0)

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                  n_actions=n_actions, name='actor')
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                    n_actions=n_actions, name='critic')

        self.target_actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,
                                         n_actions=n_actions, name='target_actor')

        self.target_critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,
                                           n_actions=n_actions, name='target_critic')

        # First time call so the target and the base networks have same parameters
        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        # Put the actor network into evaluation mode
        self.actor.eval()
        # Put state into a tensor adn move the tensor to GPU
        state = T.tensor([observation], dtype=T.float).to(self.actor.device)
        # Do a forward pass through the network -> Get the action
        mu = self.actor.forward(state).to(self.actor.device)
        # Add noise to the action
        mu_prime = mu + T.tensor(self.noise(),
                                 dtype=T.float).to(self.actor.device)
        # Put the network back into training mode
        self.actor.train()
        # Return the action. Make sure to move the action to CPU so we can read it
        return mu_prime.cpu().detach().numpy()[0]

    def remember(self, state, action, reward, state_, done):
        self.memory.store_transition(state, action, reward, state_, done)

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()

    def learn(self):
        # Can't learn if there are not enough experiences in memory
        if self.memory.mem_cntr < self.batch_size:
            return

        # Sample a random batch of experiences
        states, actions, rewards, new_state, done = \
            self.memory.sample_buffer(self.batch_size)
        # Convert everything to a tensor
        states = T.tensor(states, dtype=T.float).to(self.actor.device)
        new_states = T.tensor(new_states, dtype=T.float).to(self.actor.device)
        actions = T.tensor(actions, dtype=T.float).to(self.actor.device)
        rewards = T.tensor(rewards, dtype=T.float).to(self.actor.device)
        done = T.tensor(done).to(self.actor.device)

        ############# Critic Network #############
        # We use the target network to calculate the TD targets. 
        # If we used the regular network the target would always be moving and the algorithm would not converge.
        # So the regular network tries to predict what the real Q value is. And the target network gives is the "real" Q value.
        ##########################
        # Calculate target actions based on new_states
        target_actions = self.target_actor.forward(new_states)
        # From those new_states and target_actions calculate the Q values for next states
        target_Q_value = self.target_critic.forward(new_states, target_actions)
        # Calculate the Q values for the current states and actions with regular network
        Q_value = self.critic.forward(states, actions)

        # If the final action is done action set Q value to 0
        target_Q_value[done] = 0.0
        target_Q_value = target_Q_value.view(-1)
        # Calculate TD targets. Gamma is discount factor for future rewards
        target = rewards + self.gamma*target_Q_value
        target = target.view(self.batch_size, 1)

        # Backpropagete the critic network
        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, Q_value)
        critic_loss.backward()
        self.critic.optimizer.step()
        ############# END #############
 
        ############# Actor Network #############
        self.actor.optimizer.zero_grad()
        # We calculate the Q value using the actions that are predicted by the actor network.
        # _ since we want to maximize the Q value
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        ############# END #############

        # Lastly update the target networks
        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_params = self.critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()

        critic_state_dict = dict(critic_params)
        actor_state_dict = dict(actor_params)
        target_critic_state_dict = dict(target_critic_params)
        target_actor_state_dict = dict(target_actor_params)

        for name in critic_state_dict:
            critic_state_dict[name] = tau*critic_state_dict[name].clone() + \
                (1-tau)*target_critic_state_dict[name].clone()

        for name in actor_state_dict:
            actor_state_dict[name] = tau*actor_state_dict[name].clone() + \
                 (1-tau)*target_actor_state_dict[name].clone()

        self.target_critic.load_state_dict(critic_state_dict)
        self.target_actor.load_state_dict(actor_state_dict)
        #self.target_critic.load_state_dict(critic_state_dict, strict=False)
        #self.target_actor.load_state_dict(actor_state_dict, strict=False)
