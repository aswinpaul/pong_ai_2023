# -*- coding: utf-8 -*-
import numpy as np

m_trials = 99

d1 = []
d2 = []

for i in range(m_trials):
    file_name = 'data_dpefemethod_1_M10_' + str(i)
    with open(file_name, 'rb') as file:
        d1.append(np.load(file)) 

for i in range(m_trials):
    file_name = 'data_dpefemethod_2_M10_' + str(i)
    with open(file_name, 'rb') as file:
        d2.append(np.load(file))

data_dpefe_1 = np.array(d1, dtype = 'object')  
data_dpefe_2 = np.array(d2, dtype = 'object')  
  
with open('../data_dpefe_1_M10', 'wb') as file:
    np.save(file, data_dpefe_1)
    
with open('../data_dpefe_2_M10', 'wb') as file:
    np.save(file, data_dpefe_2)
