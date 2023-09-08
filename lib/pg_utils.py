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

        self.net = nn.Sequential(
            nn.Linear(n_states, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        
        return self.net(x)



def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            


def calc_qvals(rewards, gamma):
    res = []
    sum_r = 0.0
    for r in reversed(rewards):
        sum_r = r + gamma * sum_r
        res.append(sum_r)
    return list(reversed(res))
