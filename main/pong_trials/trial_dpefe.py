# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pongGymEnv as pongEnv

env = pongEnv.CartPoleEnv()
from pong_statetoobs import state_to_obs
env.reset(5000);

import os 
import sys


mtrial = int(sys.argv[1])
planning_horizon = int(sys.argv[2])

print(mtrial)

from pathlib import Path

path = Path(os.getcwd())
module_path = str(path.parent) + '/'
sys.path.append(module_path)

import numpy as np
import random
# agent
from pymdp.utils import random_A_matrix, random_B_matrix, random_single_categorical
from pymdp.utils import obj_array_uniform, norm_dist_obj_arr
from agents.agent_dpefe_z_learning import dpefe_agent_z

import scipy.stats as stats
import pandas as pd

random.seed(mtrial)
np.random.seed(mtrial);

# generative model of the pong-game environment

# [ self.game_state.ball0.position.x : 4-40,
# self.game_state.ball0.position.y: 8,
# self.game_state.ball0.velocity.y,
# self.game_state.paddle0.position.top.y,
# self.game_state.paddle0.position.bottom.y]

# Generative model

# (Hidden)Factors
# Ball x (Hypothesis)
s1_size = 38
# Ball y (Hypothesis)
s2_size = 8
# Pad (Hypothesis)
s3_size = 8

num_states = [s1_size, s2_size, s3_size]
num_factors = len(num_states)

# Controls
s1_actions = ['Stay', 'Play-Up', 'Play-Down']
s2_actions = ['Do nothing']
s3_actions = ['Do nothing']

num_controls = [len(s1_actions), len(s2_actions), len(s3_actions)]

# Observations
# Ball x (Hypothesis)
o1_size = 38
# Ball y (Hypothesis)
o2_size = 8
# Paddle y (Hypothesis)
o3_size = 8

num_obs = [o1_size, o2_size, o3_size]
num_modalities = len(num_obs)

####

EPS_VAL = 1e-16
A = random_A_matrix(num_obs, num_states)*0 + EPS_VAL

for i in range(s2_size):
    for j in range(s3_size):
        A[0][:,:,i,j] = np.eye(s1_size)
        
for i in range(s1_size):
    for j in range(s3_size):
        A[1][:,i,:,j] = np.eye(s2_size)
        
for i in range(s1_size):
    for j in range(s2_size):
        A[2][:,i,j,:] = np.eye(s3_size)
        
B = random_B_matrix(num_states, num_controls)*0 + EPS_VAL
B = norm_dist_obj_arr(B)

C = obj_array_uniform(num_obs)

D = obj_array_uniform(num_states)

####

EPS_VAL = 1e-16 # Negligibleconstant

#number of pong episodes in a trial
n_trials = 70
        
dpefe_agent = dpefe_agent_z(A = A,
                 B = B,
                 C = C,
                 D = D,
                 planning_horizon = planning_horizon,
                 action_precision = 1024) 

dpefe_agent.lr_pB = 1e+16

tau_trial = 0

t_length = np.zeros((n_trials, 2))
entropy_A = np.zeros((num_modalities, n_trials))
entropy_B = np.zeros((num_factors, n_trials))
entropy_C = np.zeros((num_modalities, n_trials))

for trial in range(n_trials):
        
    state = env.reset(mtrial)
    obs_list = state_to_obs(state)

    done = False
    old_reward = 0
    reward = 0
    tau = 0
    dpefe_agent.tau = 0
    
    while(done == False):
        cc = dpefe_agent.C[0]
        cd = dpefe_agent.C[1]
        ce = dpefe_agent.C[2]
        
        # Decision making
        action = dpefe_agent.step(obs_list)
        
        old_reward = reward
        n_state, reward, done, info = env.step(int(action[0]))
        
        # Inference
        prev_obs = obs_list
        obs_list = state_to_obs(n_state)
         
        hit = True if(reward > old_reward) else False
        
        if(hit):
            r = -1
            dpefe_agent.update_c(prev_obs, obs_list, reward = r, terminal = False)
        if(done):
            r = 1
            dpefe_agent.update_c(prev_obs, obs_list, reward = r, terminal = True)
            
        if(reward > 100):
            done = True
            
        tau += 1
        tau_trial += 1
        
    t_length[trial, 0] = reward
    t_length[trial, 1] = tau_trial
    
    for m in range(num_modalities):
        entropy_A[m,trial] = np.sum(stats.entropy(dpefe_agent.A[m]))
        entropy_C[m,trial] = np.sum(stats.entropy(dpefe_agent.C[m]))
        
    for f in range(num_factors):
        entropy_B[f,trial] = np.sum(stats.entropy(dpefe_agent.B[f]))
        
sep = int(tau_trial/4)
try:
    sep_trial = np.argwhere(t_length[:,1] <= sep)[-1][0] 
except:
    sep_trial = 0
sep_trial = 1 if sep_trial == 0 else sep_trial

elapse_minute_rounded = (t_length[:,1]/tau_trial)*20
    
data = {'hit_count': t_length[:, 0],
        'session_num': np.zeros(n_trials) + mtrial,
        'elapse_minute_rounded': elapse_minute_rounded.astype(int),
        'half': np.array(list(np.zeros(sep_trial)) +
                               list(np.ones(n_trials - sep_trial)))
        }

entropy_A = {'entropy_1': entropy_A[0,:],
             'entropy_2': entropy_A[1,:],
             'entropy_3': entropy_A[2,:],
            'session_num': np.zeros(n_trials) + mtrial,
            'elapse_minute_rounded': elapse_minute_rounded.astype(int),
            'half': np.array(list(np.zeros(sep_trial)) +
                                   list(np.ones(n_trials - sep_trial)))
            }

entropy_C = {'entropy_1': entropy_C[0,:],
             'entropy_2': entropy_C[1,:],
             'entropy_3': entropy_C[2,:],
            'session_num': np.zeros(n_trials) + mtrial,
            'elapse_minute_rounded': elapse_minute_rounded.astype(int),
            'half': np.array(list(np.zeros(sep_trial)) +
                                   list(np.ones(n_trials - sep_trial)))
            }

entropy_B = {'entropy_1': entropy_B[0,:],
             'entropy_2': entropy_B[1,:],
             'entropy_3': entropy_B[2,:],
            'session_num': np.zeros(n_trials) + mtrial,
            'elapse_minute_rounded': elapse_minute_rounded.astype(int),
            'half': np.array(list(np.zeros(sep_trial)) +
                                   list(np.ones(n_trials - sep_trial)))
            }

p_data = pd.DataFrame(data)

a_data = pd.DataFrame(entropy_A)
b_data = pd.DataFrame(entropy_B)
c_data = pd.DataFrame(entropy_C)

file1 = str('data_n_plot_dpefe/M') + str(planning_horizon) + str('/data_') + str(mtrial) + '.csv'
file2 = str('data_n_plot_dpefe/M') + str(planning_horizon) + str('/a_data_') +  str(mtrial) + '.csv'
file3 = str('data_n_plot_dpefe/M') + str(planning_horizon) + str('/b_data_') +  str(mtrial) + '.csv'  
file4 = str('data_n_plot_dpefe/M') + str(planning_horizon) + str('/c_data_') +  str(mtrial) + '.csv'

p_data.to_csv(file1, index=False)
a_data.to_csv(file2, index=False)
b_data.to_csv(file3, index=False)
c_data.to_csv(file4, index=False)
