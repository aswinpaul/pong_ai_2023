# -*- coding: utf-8 -*-
import pandas as pd

m_trials = 99

p_data = pd.DataFrame()

for i in range(m_trials):
    file_name = 'data_' + str(i) +'.csv'
    data = pd.read_csv(file_name)
    p_data = pd.concat([p_data, data])

a_data = pd.DataFrame()

for i in range(m_trials):
    file_name = 'a_data_' + str(i) +'.csv'
    data = pd.read_csv(file_name)
    a_data = pd.concat([a_data, data])
    
b_data = pd.DataFrame()

for i in range(m_trials):
    file_name = 'b_data_' + str(i) +'.csv'
    data = pd.read_csv(file_name)
    b_data = pd.concat([b_data, data])
    
c_data = pd.DataFrame()

for i in range(m_trials):
    file_name = 'c_data_' + str(i) +'.csv'
    data = pd.read_csv(file_name)
    c_data = pd.concat([c_data, data])
    
p_data.to_csv('../p_data_20.csv', index=False)
a_data.to_csv('../a_data_20.csv', index=False)
b_data.to_csv('../b_data_20.csv', index=False)
c_data.to_csv('../c_data_20.csv', index=False)