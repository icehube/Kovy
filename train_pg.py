#!/usr/bin/env python3

import os
import copy
import time
from datetime import datetime
import json
import pandas as pd

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter

from collections import deque

from lib.env import *
from lib.pg_utils import *


config = {
    'train_config' : {
        'episodes_to_train' : 16,
        'learning_rate' : 0.00005,
        'model_save_period' : 1000,
        'gamma' : 1,
    },

    'env_config' : {
        'budget' : 56.8,
        'min_bid' : 0.5,
        'max_bid' : 11.4,
        'n_F' : 12,
        'n_D' : 6,
        'n_G' : 2,
        'n_B' : 4,
        'team_size' : 24,
        'teams' : ['GVR', 'MAC', 'BOT', 'SHF', 'ZSK', 'LGN',
                   'SRL', 'LPT', 'HSM', 'JHN', 'VPP'],
        'agent_team' : 'BOT',
        'others_policy_model' : 'random'
    },

    'actions' : ['pass', '+0.1', '+0.5', '+1.0', '+3.0', 'max']
}


if __name__ == "__main__":
    
    run_name = 'PG_%s' % datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    train_config = config['train_config']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs('runs/%s' % run_name, exist_ok=True)
    os.makedirs('models/%s' % run_name, exist_ok=True)

    with open('runs/%s/config.json' % run_name, 'w') as f:
        json.dump(config, f, indent=4)
    with open('models/%s/config.json' % run_name, 'w') as f:
        json.dump(config, f, indent=4)
    
    env = Env('players.csv', config)

    main_model = Model_v1(env.n_states, env.n_actions).to(device)
    main_model.apply(kaiming_init)
    
    optimizer = optim.Adam(main_model.parameters(), lr=train_config['learning_rate'])
    
    loss_buffer = deque(maxlen=5000)
    reward_buffer = deque(maxlen=1000)
    
    writer = SummaryWriter('runs/%s' % run_name)

    frame_idx = 0
    
    total_rewards = []

    batch_states, batch_actions, batch_qvals = [], [], []
    cur_rewards = []
    batch_episodes = 0

    while True:
        frame_idx += 1

        state = env.state
        state_v = torch.tensor(env.state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = F.softmax(main_model(state_v), dim=1).data.cpu().numpy()[0]
        action = np.random.choice(6, p=probs)
        
        next_state, reward, done, _ = env.step(action)
        
        batch_states.append(state)
        batch_actions.append(action)
        cur_rewards.append(reward)
        
        if done:
            batch_qvals.extend(calc_qvals(cur_rewards, train_config['gamma']))
            cur_rewards.clear()
            batch_episodes += 1
            reward_buffer.append(reward)
        

        if batch_episodes < train_config['episodes_to_train']:
            continue
        
        optimizer.zero_grad()
        states_v = torch.FloatTensor(np.array(batch_states))
        batch_actions_t = torch.LongTensor(np.array(batch_actions))
        batch_qvals_v = torch.FloatTensor(np.array(batch_qvals))

        logits_v = main_model(states_v)
        log_prob_v = F.log_softmax(logits_v, dim=1)
        log_prob_actions_v = batch_qvals_v * log_prob_v[range(len(batch_states)), batch_actions_t]
        loss_v = -log_prob_actions_v.mean()
        loss_v.backward()
        clip_grad_norm_(main_model.parameters(), 5.0)
        optimizer.step()
        loss_buffer.append(loss_v.item())

        batch_episodes = 0
        batch_states.clear()
        batch_actions.clear()
        batch_qvals.clear()

        if frame_idx % train_config['model_save_period'] == 0:
            torch.save(main_model, 'models/%s/model_%d.pth' % (run_name, frame_idx / train_config['model_save_period']))

        if frame_idx % 100 == 0:
            print(frame_idx)
            writer.add_scalar('Loss', np.array(loss_buffer).mean(), frame_idx)
            writer.add_scalar('Reward', np.array(reward_buffer).mean(), frame_idx)
            
