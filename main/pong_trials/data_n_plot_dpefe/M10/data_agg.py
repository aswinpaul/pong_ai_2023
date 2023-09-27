# -*- coding: utf-8 -*-
import numpy as np

m_trials = 99
planning_horizon = 10
d1 = []
d2 = []

for i in range(m_trials):
    file_name = 'data_dpefemethod_1_M' + str(planning_horizon)+ str('_') +str(i)
    with open(file_name, 'rb') as file:
        d1.append(np.load(file)) 

for i in range(m_trials):
    file_name = 'data_dpefemethod_2_M' + str(planning_horizon)+ str('_') +str(i)
    with open(file_name, 'rb') as file:
        d2.append(np.load(file))

data_dpefe_1 = np.array(d1, dtype = 'object')  
data_dpefe_2 = np.array(d2, dtype = 'object')  
  

file_name = '../data_dpefe_1_M' + str(planning_horizon)
with open(file_name, 'wb') as file:
    np.save(file, data_dpefe_1)

file_name = '../data_dpefe_2_M' + str(planning_horizon)    
with open(file_name, 'wb') as file:
    np.save(file, data_dpefe_2)
