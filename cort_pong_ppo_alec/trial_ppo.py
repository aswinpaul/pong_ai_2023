import pongGymEnv as pongEnv

env = pongEnv.CartPoleEnv()
from pong_statetoobs import state_to_obs
env.reset(5000);

import numpy as np
import random

import pymdp 

random.seed(10)
np.random.seed(10);

memory_horizon = 1

# generative model of the pong-game environment

# [ self.game_state.ball0.position.x : 4-40,
# self.game_state.ball0.position.y: 8,
# self.game_state.ball0.velocity.y,
# self.game_state.paddle0.position.top.y,
# self.game_state.paddle0.position.bottom.y]

# o1 length: 41
# o2 length: 8
# o3 length: 8

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

import os
import numpy as np

from ppo import PPO


has_continuous_action_space = False
max_ep_len = 400
max_training_timesteps = int(1e9)
print_freq = max_ep_len * 4
log_freq = max_ep_len * 2
save_model_freq = int(2e4)

action_std = None

#update_timestep = max_ep_len * 4
K_epochs = 40
eps_clip = 0.2
gamma = 0.99

lr_actor = 0.0003
lr_critic = 0.001

random_seed = 0


pixel_dim = (6, 5)
state_dim = s1_size*100 + s2_size*10 + s3_size
action_dim = 3

log_dir = "PPO_logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_dir = log_dir + "/" + "pong" + "/"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)


i_episode = 0
print_running_reward = 0
print_running_episodes = 0

data_1 = []
data_2 = []

# training loop
m_trials = 10

for mt in range(m_trials):
    time_step = 0
    
    ppo_agent = PPO(
        state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, 
        has_continuous_action_space, action_std
    )
    
    print(mt)
    
    n_trials = 70
    tau_trial = 0
    t_length = np.zeros((n_trials, 2))
    
    for trial in range(n_trials):
        
        current_ep_reward = 0
        obs_list = []
        
        obs = env.reset(5000)
        obs = state_to_obs(obs)
        obs1 = obs[0]*100 + obs[1]*10 + obs[2]
        obs = pymdp.utils.onehot(obs1, state_dim)
    
        state = obs.reshape(state_dim)
        obs_list.append(obs)
        done = False
    
        while done == False:
            
            # select action with policy
            action = ppo_agent.select_action(state)
            
            obs, reward, done, info = env.step(action)
            
            obs = state_to_obs(obs)
            obs = obs[0]*100 + obs[1]*10 + obs[2]
            obs = pymdp.utils.onehot(obs, state_dim)
            
            state = obs.reshape(state_dim)
            obs_list.append(obs)
    
            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
    
            time_step += 1
            current_ep_reward += reward
    
            # update PPO agent
            if time_step % 100 == 0:
                ppo_agent.update()
            
            tau_trial += 1
            
            if done:
                break
            
        t_length[trial, 0] = reward
        t_length[trial, 1] = tau_trial
        
    sep = int(tau_trial/4)
    sep_trial = np.argwhere(t_length[:,1] <= sep)[-1][0]      
    sep_trial = 1 if sep_trial == 0 else sep_trial
    data_1.append(t_length[0:sep_trial, 0])
    data_2.append(t_length[sep_trial:n_trials, 0])

d_1 = np.array(data_1, dtype = 'object')
d_2 = np.array(data_2, dtype = 'object')

file1 = str('data_n_plot_ppo/') + str('data_ppo_1')
file2 = str('data_n_plot_ppo/') + str('data_ppo_2')

with open(file1, 'wb') as file:
    np.save(file, d_1)
with open(file2, 'wb') as file:
    np.save(file, d_2)