import pongGymEnv as pongEnv

env = pongEnv.CartPoleEnv()
from pong_statetoobs import state_to_obs
env.reset(5000);

import numpy as np
import random
from pymdp.agent_cl import cl_agent as agent
from pymdp.utils import random_A_matrix, random_B_matrix, obj_array_uniform, norm_dist_obj_arr   

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

#seed loops
m_trials = 40

#number of pong episodes in a trial
n_trials = 70

data_1 = []
data_2 = []

for mt in range(m_trials):
    print(mt)
        
    cl_agent = agent(A = A,
                     B = B,
                     C = C,
                     D = D,
                     memory_horizon = memory_horizon, 
                     gamma_initial = 0.55,
                     action_precision = 1024) 
    
    cl_agent.lr_pB = 1e+16
    
    tau_trial = 0
    t_length = np.zeros((n_trials, 2))
    
    for trial in range(n_trials):
        
        # print(trial)
        
        state = env.reset(mt)
        obs = state_to_obs(state)

        done = False
        old_reward = 0
        reward = 0
        tau = 0
        cl_agent.tau = 0
        
        while(done == False):
            
            # Decision making
            action = cl_agent.step(obs_list = obs)
            
            old_reward = reward
            n_state, reward, done, info = env.step(int(action[0]))
            
            # Inference
            obs = state_to_obs(n_state)
             
            hit = True if(reward > old_reward) else False
          
            # Learning
            if(hit):
                cl_agent.update_gamma(risk = -1)
                cl_agent.update_CL(tau) #state-action mapping
                
            if(done):
                cl_agent.update_gamma(risk = 1)
                cl_agent.update_CL(tau) #state-action mapping
            
            if(reward > 100):
                done = True
                
            tau += 1
            tau_trial += 1
            
        t_length[trial, 0] = reward
        t_length[trial, 1] = tau_trial
        
    sep = int(tau_trial/4)
    sep_trial = np.argwhere(t_length[:,1] <= sep)[-1][0]      
    sep_trial = 1 if sep_trial == 0 else sep_trial
    data_1.append(t_length[0:sep_trial, 0])
    data_2.append(t_length[sep_trial:n_trials, 0])

d_1 = np.array(data_1, dtype = 'object')
d_2 = np.array(data_2, dtype = 'object')

file1 = str('data_n_plot_cl/') + str('data_clmethod_1_M') + str(memory_horizon)
file2 = str('data_n_plot_cl/') + str('data_clmethod_2_M') + str(memory_horizon)

with open(file1, 'wb') as file:
    np.save(file, d_1)
with open(file2, 'wb') as file:
    np.save(file, d_2)