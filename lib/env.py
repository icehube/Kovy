import numpy as np
import pandas as pd
import torch

from lib.dqn_utils import Model_v1


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Env():

    def __init__(self, fname, config):
        
        # load data from csv file
        data = pd.read_csv(fname)
        data = data[data['Status'] != 'MINOR']
        data = data[data['Team'] != 'ENT']
        data = data[data['Team'] != 'RFA']
        data.drop(columns=['Status', 'Bid'], inplace=True)
        self.data = data.reset_index(drop=True)
        
        # config
        self.config = config['env_config']
        self.actions = config['actions']
        self.teams = config['env_config']['teams']
        self.agent_team = config['env_config']['agent_team']

        # load model for other team's policy
        if self.config['others_policy_model'] != 'random':
            self.model = torch.load(self.config['others_policy_model'])

        self.reset()
        
        self.n_actions = len(self.actions)
        self.n_states = len(self.state)


    def reset(self):
        
        #print(f"====Reset====")
        self.players = self.data.copy()
        self.initialize_status()
        while self.current_team != self.agent_team:
            self.step_one_team()
        
        self.state = self.encode()


    def initialize_status(self):

        self.done = False
        self.reward = 0
        self.i = 0
        self.current_team = self.teams[0]
        self.current_player_idx = None
        self.not_full_teams = len(self.teams)

        self.team_status = {}
        for team, group in self.players.groupby('Team'):
            if team in self.teams:
                d = {
                    'salary' : 0,
                    'F' : [],
                    'D' : [],
                    'G' : [],
                    'B' : [],
                    'available_F' : self.config['n_F'],
                    'available_D' : self.config['n_D'],
                    'available_G' : self.config['n_G'],
                    'available_B' : self.config['n_B'],
                    'available' : self.config['team_size'],
                    'passed' : False
                }
                for idx, player in group.iterrows():
                    pos = player['Pos']
                    d['salary'] += player['Salary']
                    if d[f'available_{pos}'] > 0:
                        d[pos].append(idx)
                        d[f'available_{pos}'] -= 1
                    else:
                        d['B'].append(idx)
                        d['available_B'] -= 1
                    d['available'] -= 1
                d['budget'] = self.config['budget'] - d['salary'] - d['available'] * self.config['min_bid']
                self.team_status[team] = d
        for team, status in self.team_status.items():
            if status['available'] == 0:
                status['passed'] = True
                self.not_full_teams -= 1


    def step(self, action_idx):

        #print(f"====Step====")
        #print(f"Action received: {self.actions[action_idx]}")
        self.step_one_team(action_idx)
        while not self.done:
            self.step_one_team()
            if self.current_team == self.agent_team and self.team_status[self.agent_team]['passed'] == False:
                break
        
        state = self.encode()
        reward = self.reward
        done = self.done
        self.i += 1
        if self.done:
            # print(self.reward)
            # print(self.i)
            self.i = 0
            self.reset()            
        else:
            self.state = state

        return state, reward, done, None


    def step_one_team(self, action_idx = None):
        
        status = self.team_status[self.current_team]
        if not status['passed']:
            if self.current_player_idx == None:
                # introducting player
                s = self.config['n_F'] + self.config['n_D'] + self.config['n_G']
                probs = [self.config['n_F'] / s, self.config['n_D'] / s, self.config['n_G'] / s]
                while True:
                    pos = ['F', 'D', 'G'][np.random.choice(3, p=probs)]
                    if status[f'available_{pos}'] + status['available_B'] > 0:
                        break
                players = self.players[(self.players['Pos'] == pos) & (self.players['Team'] == 'UFA')].sort_values(by='Pts', ascending=False)
                self.current_player_idx = np.random.choice(players.index[:3*self.config[f'n_{pos}']])
                self.current_high_bid = self.config['min_bid']
                self.highest_bidder = self.current_team
                self.not_passed_teams = 0
                for team in self.teams:
                    if self.team_status[team]['available'] > 0:
                        self.team_status[team]['passed'] = False
                        self.not_passed_teams += 1
                if self.not_passed_teams == 1:
                    self.players.loc[self.current_player_idx, 'Team'] = self.highest_bidder
                    self.players.loc[self.current_player_idx, 'Salary'] = self.current_high_bid
                    #print(f"{self.highest_bidder} won the player")
                    self.not_passed_teams = 0
                    for team in self.teams:
                        if self.team_status[team]['available'] != 0:
                            self.team_status[team]['passed'] = False
                            self.not_passed_teams += 1
                    status = self.team_status[self.highest_bidder]
                    status['salary'] += self.current_high_bid
                    status['budget'] -= self.current_high_bid - self.config['min_bid']
                    if status[f'available_{pos}'] == 0:
                        m = self.players.loc[self.current_player_idx]['Pts']
                        i = self.current_player_idx
                        for idx in status[pos]:
                            if m > self.players.loc[idx]['Pts']:
                                m = self.players.loc[idx]['Pts']
                                i = idx
                        if i != self.current_player_idx:
                            status[pos].remove(i)
                            status[pos].append(self.current_player_idx)
                            status['B'].append(i)
                        else:
                            status['B'].append(self.current_player_idx)
                        status['available_B'] -= 1
                        status['available'] -= 1
                    else:
                        status[pos].append(self.current_player_idx)
                        status[f'available_{pos}'] -= 1
                        status['available'] -= 1
                    if status['available'] == 0:
                        #print(f'team {self.highest_bidder} full')
                        status['passed'] = True
                        self.not_full_teams -= 1
                    if self.not_full_teams == 0:
                        #print('all team full')
                        self.done = True
                        self.calculate_reward()
                    self.current_player_idx = None

                #print(f"{self.current_team} introduced")
            else:
                # bidding player
                pos = self.players.loc[self.current_player_idx]['Pos']
                if action_idx == None:
                    action_idx = self.get_action()
                action = self.actions[action_idx]
                if action == 'pass':
                    bid = 0
                elif action == 'max':
                    bid = self.config['max_bid']
                else:
                    bid = self.current_high_bid + float(action)
                bid = min(bid, status['budget'] + self.config['min_bid'])

                if status[f'available_{pos}'] + status['available_B'] == 0:
                    bid = 0

                if bid <= self.current_high_bid:
                    #print(f"{self.current_team} passed")
                    status['passed'] = True
                    self.not_passed_teams -= 1
                    if self.not_passed_teams == 1:
                        self.players.loc[self.current_player_idx, 'Team'] = self.highest_bidder
                        self.players.loc[self.current_player_idx, 'Salary'] = self.current_high_bid
                        #print(f"{self.highest_bidder} won the player")
                        self.not_passed_teams = 0
                        for team in self.teams:
                            if self.team_status[team]['available'] != 0:
                                self.team_status[team]['passed'] = False
                                self.not_passed_teams += 1
                        status = self.team_status[self.highest_bidder]
                        status['salary'] += self.current_high_bid
                        status['budget'] -= self.current_high_bid - self.config['min_bid']
                        if status[f'available_{pos}'] == 0:
                            m = self.players.loc[self.current_player_idx]['Pts']
                            i = self.current_player_idx
                            for idx in status[pos]:
                                if m > self.players.loc[idx]['Pts']:
                                    m = self.players.loc[idx]['Pts']
                                    i = idx
                            if i != self.current_player_idx:
                                status[pos].remove(i)
                                status[pos].append(self.current_player_idx)
                                status['B'].append(i)
                            else:
                                status['B'].append(self.current_player_idx)
                            status['available_B'] -= 1
                            status['available'] -= 1
                        else:
                            status[pos].append(self.current_player_idx)
                            status[f'available_{pos}'] -= 1
                            status['available'] -= 1
                        if status['available'] == 0:
                            #print(f'team {self.highest_bidder} full')
                            status['passed'] = True
                            self.not_full_teams -= 1
                        if self.not_full_teams == 0:
                            #print('all team full')
                            self.done = True
                            self.calcuate_reward()
                        self.current_player_idx = None

                else:
                    #print(f"{self.current_team} bidding {bid}")
                    self.current_high_bid = bid
                    self.highest_bidder = self.current_team
            
        # next team
        idx = self.teams.index(self.current_team)
        idx = (idx + 1) % len(self.teams)
        self.current_team = self.teams[idx]


    def calculate_reward(self):
        other_team_points = []
        for team, group in self.players.groupby('Team'):
            if team in self.teams and team != self.agent_team:
                s = 0
                for pos in ['F', 'D', 'G']:
                    s += group[group['Pos'] == pos].nlargest(self.config[f'n_{pos}'], 'Pts')['Pts'].mean()
                other_team_points.append(s)
                    
            elif team == self.agent_team:
                t = 0
                for pos in ['F', 'D', 'G']:
                    t += group[group['Pos'] == pos].nlargest(self.config[f'n_{pos}'], 'Pts')['Pts'].mean()
        self.reward = t - sum(other_team_points) / len(other_team_points)


    def get_action(self):

        if self.config['others_policy_model'] == 'random':
            probs = [0.3, 0.4, 0.2, 0.05, 0.03, 0.02]
            return np.random.choice(6, p=probs)
        else:
            state = self.encode()
            state_v = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                action = self.model(state_v).detach().cpu().numpy().argmax()
            return action
        
    
    def encode(self):
        
        index = self.teams.index(self.current_team)
        teams = []
        teams.extend(self.teams[index:])
        teams.extend(self.teams[:index])
        state = []
        for team in teams:
            status = self.team_status[team]
            if status['available'] == 0:
                state.append(0)
            else:
                state.append(status['budget'] / status['available'])
            state.append(status['available_F'] / self.config['n_F'])
            state.append(status['available_D'] / self.config['n_D'])
            state.append(status['available_G'] / self.config['n_G'])
            state.append(status['available_B'] / self.config['n_B'])
            state.append(int(status['passed']))
            for pos in ['F', 'D', 'G']:
                if len(status[pos]) == 0:
                    state.append(0)
                    state.append(0)
                else:
                    pts = self.players.loc[status[pos]]['Pts']
                    state.append(pts.mean() / 50)
                    state.append(pts.min() / 50)
            
        for pos in ['F', 'D', 'G']:
            players = self.players[(self.players['Team'] == 'UFA') & (self.players['Pos'] == pos)]
            a = players.sort_values(by='Pts', ascending=False)
            state.append(a.iloc[0]['Pts'] / 50)
            state.append(a.iloc[self.config[f'n_{pos}']]['Pts'] / 50)
            state.append(a.iloc[2*self.config[f'n_{pos}']]['Pts'] / 50)
            state.append(a.iloc[3*self.config[f'n_{pos}']]['Pts'] / 50)
            
        if self.current_player_idx != None:
            player = self.players.loc[self.current_player_idx]
            if player['Pos'] == 'F': state.extend([1, 0, 0])
            elif player['Pos'] == 'D': state.extend([0, 1, 0])
            elif player['Pos'] == 'G': state.extend([0, 0, 1])
            else: raise Exception
            state.append(player['Pts'] / 50)
            state.append(self.current_high_bid / 50)
        else:
            state.extend([0] * 5)

        return np.array(state, dtype=np.float32)

