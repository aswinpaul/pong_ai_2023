#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 14:35:02 2022

@author: aswinpaul
"""
# Core modules
import numpy as np

# Environment
import pongGymEnv as pongEnv
env = pongEnv.CartPoleEnv()
env.reset(5000);
from pong_statetoobs import state_to_obs


# agent
import agent_soph_inf_learnc as helper
from agent_soph_inf_learnc import agent as dpefe_agent

# %% 
# Generative model

# (Hidden)Factors
# Ball x (Hypothesis)
s1_size = 41
# Ball y (Hypothesis)
s2_size = 9
# Paddle y (Hypothesis)
s3_size = 9

num_states = [s1_size, s2_size, s3_size]
num_factors = len(num_states)

# Controls
s1_actions = ['Do nothing']
s2_actions = ['Do nothing']
s3_actions = ['Stay', 'Play-Up', 'Play-Down']

num_controls = [len(s1_actions), len(s2_actions), len(s3_actions)]

# Observations
# Ball x (Hypothesis)
o1_size = 41
# Ball y (Hypothesis)
o2_size = 9
# Paddle y (Hypothesis)
o3_size = 9

num_obs = [o1_size, o2_size, o3_size]
num_modalities = len(num_obs)

# %%

T = 2

m_trials = 10
n_trials = 70

data_1 = []
data_2 = []

for mt in range(m_trials):
    print(mt)

    A = helper.random_A_matrix(num_obs, num_states)*0 + 1e-10
    
    for i in range(s2_size):
        for j in range(s3_size):
            A[0][:,:,i,j] = np.eye(s1_size)
            
    for i in range(s1_size):
        for j in range(s3_size):
            A[1][:,i,:,j] = np.eye(s2_size)
            
    for i in range(s1_size):
        for j in range(s2_size):
            A[2][:,i,j,:] = np.eye(s3_size)
            
    a = dpefe_agent(num_states, num_obs, num_controls, planning_horizon = T, 
                    a = A, MDP = True, threshold = 1/2)
    
    tau_trial = 0
    t_length = np.zeros((n_trials, 2))
    for trial in range(n_trials):
        
        print("Trial", trial)
        state = env.reset(mt)
        obs = state_to_obs(state)
        prev_obs = obs
        reward = 0
        old_reward = 0
        
        done = False
        tau = 0
        while(done == False):
            
            action = a.step(obs, tau)
            old_reward = reward
            n_state, reward, done, info = env.step(action)
            prev_obs = obs
            obs = state_to_obs(n_state)
            
            hit = True if(reward > old_reward) else False

            if(hit):
                rew = 1
                a.update_c(prev_obs, obs, rew)
            if(done):
                rew = -1
                a.update_c(prev_obs, obs, rew)
            else:
                rew = 0
                a.update_c(prev_obs, obs, rew)

            tau += 1
            tau_trial += 1
            
        t_length[trial, 0] = reward
        t_length[trial, 1] = tau_trial
        a.end_of_trial()
        
    sep = tau_trial/4
    sep_trial = np.argwhere(t_length[:,1] <= sep)[-1][0]      
    
    data_1 = data_1 + list(t_length[0:sep_trial, 0])
    data_2 = data_2 + list(t_length[sep_trial:n_trials, 0])

d_1 = np.array(data_1)
d_2 = np.array(data_2)

with open('data_soph_inf_1.npy', 'wb') as file:
    np.save(file, d_1)
with open('data_soph_inf_2.npy', 'wb') as file:
    np.save(file, d_2)