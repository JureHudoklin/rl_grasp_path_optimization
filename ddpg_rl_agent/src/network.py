

import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='checkpoints/'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        try:
            os.makedirs(self.checkpoint_dir)
            print("Directory: ", self.checkpoint_dir, ". Created")
        except FileExistsError:
            print("Directory: ", self.checkpoint_dir, ". Already exists")
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        # 2 Fully connected layers
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)

        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,
                                    weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        """ 
        Forward pass of critic network. Takes in an action predicted by 
        actor network to calculate the Q value for the given state-action pair.
        --------
        Arguments:
            state: state of the environment
            action: action predicted by actor network
        --------
        Returns:
            state_action_value: Q value for the given state-action pair
        """
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)

        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)

        action_value = self.action_value(action)

        state_action_value = F.relu(T.add(state_value, action_value))

        state_action_value = self.q(state_action_value)
        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        # create model directory
        state = {'state_dict': self.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        T.save(state, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        loaded_data = T.load(self.checkpoint_file)
        self.load_state_dict(loaded_data['state_dict'])
        self.optimizer.load_state_dict(loaded_data['optimizer'])

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        # create model directory
        T.save(self.state_dict(), checkpoint_file)


class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='checkpoints/'):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        try:
            os.makedirs(self.checkpoint_dir)
            print("Directory: ", self.checkpoint_dir, ". Created")
        except FileExistsError:
            print("Directory: ", self.checkpoint_dir, ". Already exists")
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        #self.bn1 = nn.BatchNorm1d(self.fc1_dims)
        #self.bn2 = nn.BatchNorm1d(self.fc2_dims)

        self.mu = nn.Linear(self.fc2_dims, self.n_actions)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.mu.weight.data.uniform_(-f3, f3)
        self.mu.bias.data.uniform_(-f3, f3)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        """
        Forward pass for actor network. Based on the state calculates the action.
        Because it is a deterministic alghorithm the output is directly the action and not the distribution.
        --------
        Arguments:
            state: state of the environment
        --------
        Returns:
            action: action for the given state
        """
        # First FC
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        # Second FC
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # Output scaled to [-1, 1]
        x = T.tanh(self.mu(x))/5
        return x

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        # create model directory
        state = {'state_dict': self.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        T.save(state, self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        loaded_data = T.load(self.checkpoint_file)
        self.load_state_dict(loaded_data['state_dict'])
        self.optimizer.load_state_dict(loaded_data['optimizer'])

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        # create model directory
        T.save(self.state_dict(), checkpoint_file)
