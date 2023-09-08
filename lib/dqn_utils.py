import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


class Model_v1(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Model_v1, self).__init__()

        self.common = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.dropout = nn.Dropout(p=0.5)
        self.fc_adv = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions)
        )
        self.fc_val = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        
        x = self.common(x)
        x = self.dropout(x)
        val = self.fc_val(x)
        adv = self.fc_adv(x)
        return val + (adv - adv.mean(dim=1, keepdim=True))


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            


class ExperienceReplayBuffer():

    def __init__(self, buffer_size):

        self.buffer_size = buffer_size
        self.state_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.done_buffer = []
        self.next_state_buffer = []
        self.pos = 0

    def add(self, state, action, reward, done, next_state):

        if len(self.state_buffer) < self.buffer_size:
            self.state_buffer.append(state)
            self.action_buffer.append(action)
            self.reward_buffer.append(reward)
            self.done_buffer.append(done)
            self.next_state_buffer.append(next_state)
        else:
            self.state_buffer[self.pos] = state
            self.action_buffer[self.pos] = action
            self.reward_buffer[self.pos] = reward
            self.done_buffer[self.pos] = done
            self.next_state_buffer[self.pos] = next_state

        self.pos = (self.pos + 1) % self.buffer_size


    def sample(self, n):

        idxs = np.random.choice(len(self.state_buffer), n)
        states = np.array([self.state_buffer[i] for i in idxs])
        actions = np.array([self.action_buffer[i] for i in idxs])
        rewards = np.array([self.reward_buffer[i] for i in idxs])
        dones = np.array([self.done_buffer[i] for i in idxs], dtype=np.bool_)
        next_states = np.array([self.next_state_buffer[i] for i in idxs])
        
        return states, actions, rewards, dones, next_states
    

