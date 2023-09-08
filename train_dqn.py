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
from lib.dqn_utils import *


config = {
    'train_config' : {
        'replay_buffer_size' : 100000,
        'train_start' : 20000,
        'target_model_sync_period' : 5000,
        'target_model_sync_alpha' : 0.5,
        'pop_n_per_step' : 64,
        'epsilon_frames' : 500000,
        'epsilon_start' : 1.0,
        'epsilon_final' : 0.1,
        'learning_rate' : 0.00003,
        'gamma' : 0.99,
        'batch_size' : 512,
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
    
    run_name = 'DQN_%s' % datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
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
    target_model = copy.deepcopy(main_model)
    
    buffer = ExperienceReplayBuffer(buffer_size=train_config['replay_buffer_size'])
    optimizer = optim.Adam(main_model.parameters(), lr=train_config['learning_rate'])
    
    loss_buffer = deque(maxlen=5000)
    reward_buffer = deque(maxlen=1000)
    state_v_buffer = deque(maxlen=1000)
    
    writer = SummaryWriter('runs/%s' % run_name)

    frame_idx = 0
    
    while True:
        frame_idx += 1

        epsilon = max(train_config['epsilon_final'], train_config['epsilon_start'] - frame_idx / train_config['epsilon_frames'])

        state = env.state
        state_v = torch.tensor(env.state, dtype=torch.float32).unsqueeze(0).to(device)
        if np.random.rand() < epsilon:
            probs = [0.3, 0.4, 0.2, 0.05, 0.03, 0.02]
            action = np.random.choice(6, p=probs)
        else:
            with torch.no_grad():
               action = main_model(state_v).detach().cpu().numpy().argmax()
        next_state, reward, done, _ = env.step(action)
        buffer.add(state, action, reward, done, next_state)
        if done: reward_buffer.append(reward)
        
        if len(buffer.state_buffer) < train_config['train_start'] or frame_idx % train_config['pop_n_per_step'] != 0:
            continue
        
        optimizer.zero_grad()
        states, actions, rewards, dones, next_states = buffer.sample(train_config['batch_size'])

        states_v = torch.tensor(states, dtype=torch.float32).to(device)
        next_states_v = torch.tensor(next_states, dtype=torch.float32).to(device)
        actions_v = torch.tensor(actions, dtype=torch.int64).to(device)
        rewards_v = torch.tensor(rewards, dtype=torch.float32).to(device)
        done_mask = torch.tensor(dones, dtype=torch.bool).to(device)

        state_action_values = main_model(states_v).gather(0, actions_v.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_actions = main_model(next_states_v).max(1)[1]
            next_state_values = target_model(next_states_v).gather(0, next_state_actions.unsqueeze(-1)).squeeze(-1)
            next_state_values[done_mask] = 0.0
            expected_state_action_values = next_state_values.detach() * train_config['gamma'] + rewards_v
        loss_v = nn.MSELoss()(state_action_values, expected_state_action_values)
        loss_v.backward()
        clip_grad_norm_(main_model.parameters(), 5.0)
        optimizer.step()
        loss_buffer.append(loss_v.item())
        state_v_buffer.append(state_action_values.mean().item())

        if frame_idx % train_config['target_model_sync_period'] == 0:
            state = main_model.state_dict()
            tgt_state = target_model.state_dict()
            for k, v in state.items():
                tgt_state[k] = tgt_state[k] * train_config['target_model_sync_alpha'] + (1 - train_config['target_model_sync_alpha']) * v
            target_model.load_state_dict(tgt_state)
            torch.save(main_model, 'models/%s/model_%d.pth' % (run_name, frame_idx / train_config['target_model_sync_period']))

        if frame_idx % 100 == 0:
            print(frame_idx)
            writer.add_scalar('Epsilon', epsilon, frame_idx)
            writer.add_scalar('Loss', np.array(loss_buffer).mean(), frame_idx)
            writer.add_scalar('Reward', np.array(reward_buffer).mean(), frame_idx)
            writer.add_scalar('State_value', np.array(state_v_buffer).mean(), frame_idx)

