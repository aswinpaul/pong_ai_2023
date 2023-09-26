import pongGymEnv as pongEnv
env = pongEnv.CartPoleEnv()
from pong_statetoobs import state_to_obs
env.reset(5000);

import numpy as np
import random

m_trials = 114
n_trials = 70

data_1 = []
data_2 = []

for mt in range(m_trials):
    print(mt)
    
    random.seed(mt)
    np.random.seed(mt);
    
    tau = 0
    t_length = np.zeros((n_trials, 2))
    
    for trial in range(n_trials):
        
        # print(trial)
        state = env.reset(mt)
        obs = state_to_obs(state)
        
        done = False
        while(done == False):
            
            # Decision making
            action = np.random.randint(0,3)
            n_state, reward, done, info = env.step(action)
            obs = state_to_obs(n_state)
            
            if(done):
                break
            tau += 1
            
        t_length[trial, 0] = reward
        t_length[trial, 1] = tau
    
    sep = tau/4
    sep_trial = np.argwhere(t_length[:,1] <= sep)[-1][0]      
    
    data_1.append(t_length[0:sep_trial, 0])
    data_2.append(t_length[sep_trial:n_trials, 0])

d_1 = np.array(data_1, dtype = 'object')
d_2 = np.array(data_2, dtype = 'object')

with open('data_random_agent_1', 'wb') as file:
    np.save(file, d_1)
with open('data_random_agent_2', 'wb') as file:
    np.save(file, d_2)