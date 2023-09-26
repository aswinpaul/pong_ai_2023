#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:35:02 2022

@author: aswinpaul
"""
import time
# Start time of program
st = time.process_time()

# Core modules
import numpy as np
from matplotlib import pyplot as plt


# Environment
import pongGymEnv as pongEnv
env = pongEnv.CartPoleEnv()
env.reset(5000);
from pong_statetoobs import state_to_obs


# agent
import agent_dpefe as helper
from agent_dpefe import agent as dpefe_agent

# %% 
# Generative model

# (Hidden)Factors
# Ball x (Hypothesis)
s1_size = 41
# Ball y (Hypothesis)
s2_size = 9

num_states = [s1_size, s2_size]
num_factors = len(num_states)

# Rewards
reward_modes = 3 #Max score-5 (assumption)

# Controls
s1_actions = ['Do nothing']
s2_actions = ['Stay', 'Play-Up', 'Play-Down']

num_controls = [len(s1_actions), len(s2_actions)]

# Observations
# Ball x (Hypothesis)
o1_size = 41
# Ball y (Hypothesis)
o2_size = 9

num_obs = [o1_size, o2_size]
num_modalities = len(num_obs)

# %%

T = 30
m_trials = 10
n_trials = 72
horizon = 10000

t_length = np.zeros((m_trials, n_trials))
rally_length = np.zeros((m_trials, n_trials))

for mt in range(m_trials):
    print(mt)

    A = helper.random_A_matrix(num_obs, num_states)*0 + 1e-10
    
    for i in range(s2_size):
        A[0][:,:,i] = np.eye(s1_size)
            
    for j in range(s1_size):       
        A[1][:,j,:] = np.eye(s2_size)
    
    
    a = dpefe_agent(num_states, num_obs, num_controls, planning_horizon = T, 
                    a = A, MDP = True)

    for trial in range(n_trials):
        
        print("Trial", trial)
        state = env.reset(mt)
        obs = state_to_obs(state)
        
        reward = 0
        old_reward = 0
        
        for t in range(horizon):
            
            action = a.step(obs, t)
            old_reward = reward
            n_state, reward, done, info = env.step(action)
            obs = state_to_obs(n_state)
            
            hit = True if(reward > old_reward) else False

            if(hit):
                a.update_c(obs)

            if(done):
                break
                
        rally_length[mt,trial] = reward
        t_length[mt,trial] = t
        a.end_of_trial()

# %%

score_raw = rally_length
episodes = n_trials

score_1_raw = score_raw[:, 0:int(episodes/4)]
score_2_raw = score_raw[:, int(episodes/4):episodes]

score_length_1 = np.mean(score_1_raw, axis = 0)
score_length_2 = np.mean(score_2_raw, axis = 0)
x = [score_length_1, score_length_2]

#Plotting
xy = [x[0], x[1]]

plt.boxplot(xy, showmeans=True, showfliers=False, positions=[1,1.2])
plt.ylabel("Average hit count")
plt.xlabel("Session-1 and Session-2")
plt.title("Game play")
#plt.ylim(bottom=0)

# %%

# plt.savefig('CL method agent.png', dpi=500, bbox_inches='tight')

with open('DPEFE.npy', 'wb') as file:
    np.save(file, rally_length)

# %%

# T test

# import scipy

# aa = np.mean(score_1_raw, axis=1)
# bb = np.mean(score_2_raw, axis=1)

# scipy.stats.ttest_ind(aa, bb)

# %%

# get execution time
# get the end time
et = time.process_time()

res = et - st
print('CPU Execution time:', res, 'seconds')