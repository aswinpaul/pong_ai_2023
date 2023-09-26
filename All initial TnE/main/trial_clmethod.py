import pongGymEnv as pongEnv
env = pongEnv.CartPoleEnv()
from pong_statetoobs import state_to_obs
env.reset(5000);

import numpy as np
import random
import agent_clmethod as helper
from agent_clmethod import agent   

random.seed(10)
np.random.seed(10);

# [ self.game_state.ball0.position.x : 4-40,
# self.game_state.ball0.position.y: 8,
# self.game_state.ball0.velocity.y,
# self.game_state.paddle0.position.top.y,
# self.game_state.paddle0.position.bottom.y]

# o1 length: 41
# o2 length: 9
# o3 length: 9

# Generative model

# (Hidden)Factors
# Ball x (Hypothesis)
s1_size = 41
# Ball y (Hypothesis)
s2_size = 9
# Pad (Hypothesis)
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

EPS_VAL = 1e-16 # Negligibleconstant

m_trials = 40
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
    
    memory_horizon = 12
    cl_agent = agent(num_states, num_obs, num_controls, a = A, 
                     horizon=memory_horizon, gamma_initial = 0.9)    
    
    tau_trial = 0
    t_length = np.zeros((n_trials, 2))
    for trial in range(n_trials):
        
        # print(trial)
        
        state = env.reset(mt)
        obs = state_to_obs(state)
        cl_agent.infer_hiddenstate(obs)

        done = False
        old_reward = 0
        reward = 0
        tau = 0
        
        while(done == False):
            
            # Decision making
            action = cl_agent.take_decision()
            
            old_reward = reward
            n_state, reward, done, info = env.step(action)
            
            # Inference
            obs = state_to_obs(n_state)
            cl_agent.infer_hiddenstate(obs)
            
            hit = True if(reward > old_reward) else False
          
            # Learning
            if(hit):
                cl_agent.update_gamma(hit=hit, miss=False)
                cl_agent.update_C(tau) #state-action mapping
                
            if(done):
                cl_agent.update_gamma(miss=True, hit=False)
                cl_agent.update_C(tau) #state-action mapping
            
            tau += 1
            tau_trial += 1
            
        t_length[trial, 0] = reward
        t_length[trial, 1] = tau_trial
        
    sep = tau_trial/4
    sep_trial = np.argwhere(t_length[:,1] <= sep)[-1][0]      
    
    data_1.append(t_length[0:sep_trial, 0])
    data_2.append(t_length[sep_trial:n_trials, 0])

d_1 = np.array(data_1, dtype = 'object')
d_2 = np.array(data_2, dtype = 'object')

with open('data_clmethod_1', 'wb') as file:
    np.save(file, d_1)
with open('data_clmethod_2', 'wb') as file:
    np.save(file, d_2)