#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 08:43:51 2024

@author: aswinpaul
"""

from matplotlib import pyplot as plt
import numpy as np


#%% Data from simulations

plot_raw = False
data = {}
trials = 40
episodes = 70

with open('data_clmethod_1_M1', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M1', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)

sl = []
for i in range(trials):
    sl.append(list(score_length_1[i]) + list(score_length_2[i]))
    
data_1 = np.array(sl, dtype = 'object')

data[0] = data_1

with open('data_clmethod_1_M2', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M2', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)

sl = []
for i in range(trials):
    sl.append(list(score_length_1[i]) + list(score_length_2[i]))
    
data_1 = np.array(sl, dtype = 'object')

data[1] = data_1


with open('data_clmethod_1_M3', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M3', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)

sl = []
for i in range(trials):
    sl.append(list(score_length_1[i]) + list(score_length_2[i]))
    
data_1 = np.array(sl, dtype = 'object')

data[2] = data_1

with open('data_clmethod_1_M4', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M4', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)

sl = []
for i in range(trials):
    sl.append(list(score_length_1[i]) + list(score_length_2[i]))
    
data_1 = np.array(sl, dtype = 'object')

data[3] = data_1

agents = 4
episodes = 70

for i in range(agents):
    data[i] = np.array(data[i], dtype=int)
    data[i] = np.clip(data[i], 0, 5)


sample = np.shape(data[0][:,:][0])[0]

data_mean = {}
data_fit = {}  
x_vals = np.linspace(0,69,70)

for i in range(agents):
    data_mean[i] = np.mean(data[i][:,:], axis=0)
    m,b = np.polyfit(x_vals, data_mean[i], 1)
    data_fit[i] = m*x_vals + b

    plt.plot(range(sample-1), data_fit[i][:-1])


plt.legend(["CL agent (T = 1)",
            "CL agent (T = 2)",
            "CL agent (T = 3)",
            "CL agent (T = 4)"
            ])

plt.xlabel("Episode number")
plt.ylabel("Avg. rally length")

plt.savefig('fit_graph.png', dpi=500, bbox_inches='tight')

# data_std = {}    
# for i in range(agents):
#     data_std[i] = np.std(data[i][:,:], axis=0)
    
#     plt.fill_between(range(sample-1), 
#                       data_mean[i][:-1] + data_std[i][:-1],
#                       data_mean[i][:-1] - data_std[i][:-1],
#                       alpha=0.3)
    
#     plt.ylim(0,None)