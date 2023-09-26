#!/usr/bin/env python
# coding: utf-8

# In[1]:


from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats


#%% Data from simulations

plot_raw = False

with open('data_clmethod_1_M1', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M1', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)
 
if(plot_raw == False):
    mean_1 = []
    for i in range(len(score_length_1)):
        mean_1.append(np.mean(score_length_1[i]))
    mean_2 = []
    for i in range(len(score_length_2)):
        mean_2.append(np.mean(score_length_2[i]))
        
    x = [mean_1, mean_2]
    
else:
    raw_1 = []
    for i in range(len(score_length_1)):
        raw_1 = raw_1 + list(score_length_1[i])
        
    raw_2 = []
    for i in range(len(score_length_2)):
        raw_2 = raw_2 + list(score_length_2[i])
        
    x = [raw_1, raw_2]
    
with open('data_clmethod_1_M2', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M2', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)
 
if(plot_raw == False):
    mean_1 = []
    for i in range(len(score_length_1)):
        mean_1.append(np.mean(score_length_1[i]))
    mean_2 = []
    for i in range(len(score_length_2)):
        mean_2.append(np.mean(score_length_2[i]))
        
    xx = [mean_1, mean_2]
    
else:
    raw_1 = []
    for i in range(len(score_length_1)):
        raw_1 = raw_1 + list(score_length_1[i])
        
    raw_2 = []
    for i in range(len(score_length_2)):
        raw_2 = raw_2 + list(score_length_2[i])
        
    xx = [raw_1, raw_2]

with open('data_clmethod_1_M3', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M3', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)

if(plot_raw == False):
    mean_1 = []
    for i in range(len(score_length_1)):
        mean_1.append(np.mean(score_length_1[i]))
    mean_2 = []
    for i in range(len(score_length_2)):
        mean_2.append(np.mean(score_length_2[i]))
        
    y = [mean_1, mean_2]
    
else:
    raw_1 = []
    for i in range(len(score_length_1)):
        raw_1 = raw_1 + list(score_length_1[i])
        
    raw_2 = []
    for i in range(len(score_length_2)):
        raw_2 = raw_2 + list(score_length_2[i])
        
    y = [raw_1, raw_2]

with open('data_clmethod_1_M4', 'rb') as file:
    score_length_1 = np.load(file, allow_pickle=True)
    
with open('data_clmethod_2_M4', 'rb') as file:
    score_length_2 = np.load(file, allow_pickle=True)
    
if(plot_raw == False):
    mean_1 = []
    for i in range(len(score_length_1)):
        mean_1.append(np.mean(score_length_1[i]))
    mean_2 = []
    for i in range(len(score_length_2)):
        mean_2.append(np.mean(score_length_2[i]))
        
    z = [mean_1, mean_2]
    
else:
    raw_1 = []
    for i in range(len(score_length_1)):
        raw_1 = raw_1 + list(score_length_1[i])
        
    raw_2 = []
    for i in range(len(score_length_2)):
        raw_2 = raw_2 + list(score_length_2[i])
        
    z = [raw_1, raw_2]

#%% Data from paper

df = pd.read_csv('arl_data_kagan22.csv', sep=',',header=None)

mcc_1 = np.array(df[1][2:8])
mcc_2 = np.array(df[3][2:8])
hcc_1 = np.array(df[5][2:8])
hcc_2 = np.array(df[7][2:8])
is_1 = np.array(df[9][2:8])
is_2 = np.array(df[11][2:8])
     
min_val_1 = float(mcc_1[0])
p10_1 = float(mcc_1[1])
median_1 = float(mcc_1[2])
average_1 = float(mcc_1[3])
p90_1 = float(mcc_1[4])
max_val_1 = float(mcc_1[5])

min_val_2 = float(mcc_2[0])
p10_2 = float(mcc_2[1])
median_2 = float(mcc_2[2])
average_2 = float(mcc_2[3])
p90_2 = float(mcc_2[4])
max_val_2 = float(mcc_2[5])

min_val_3 = float(hcc_1[0])
p10_3 = float(hcc_1[1])
median_3 = float(hcc_1[2])
average_3 = float(hcc_1[3])
p90_3 = float(hcc_1[4])
max_val_3 = float(hcc_1[5])

min_val_4 = float(hcc_2[0])
p10_4 = float(hcc_2[1])
median_4 = float(hcc_2[2])
average_4 = float(hcc_2[3])
p90_4 = float(hcc_2[4])
max_val_4 = float(hcc_2[5])

min_val_5 = float(is_1[0])
p10_5 = float(is_1[1])
median_5 = float(is_1[2])
average_5 = float(is_1[3])
p90_5 = float(is_1[4])
max_val_5 = float(is_1[5])

min_val_6 = float(is_2[0])
p10_6 = float(is_2[1])
median_6 = float(is_2[2])
average_6 = float(is_2[3])
p90_6 = float(is_2[4])
max_val_6 = float(is_2[5])

stats1 = [{'mean': average_1, 'med': median_1, 'q1': p10_1, 'q3': p90_1, 'whislo': min_val_1, 'whishi': max_val_1},
          {'mean': average_2, 'med': median_2, 'q1': p10_2, 'q3': p90_2, 'whislo': min_val_2, 'whishi': max_val_2},
          {'mean': average_3, 'med': median_3, 'q1': p10_3, 'q3': p90_3, 'whislo': min_val_3, 'whishi': max_val_3},
          {'mean': average_4, 'med': median_4, 'q1': p10_4, 'q3': p90_4, 'whislo': min_val_4, 'whishi': max_val_4},
          {'mean': average_5, 'med': median_5, 'q1': p10_5, 'q3': p90_5, 'whislo': min_val_5, 'whishi': max_val_5},
          {'mean': average_6, 'med': median_6, 'q1': p10_6, 'q3': p90_6, 'whislo': min_val_6, 'whishi': max_val_6}]



# %% Box Plot

#%% Plotting

# Paper 
fig, ax = plt.subplots()
p = [0.3, 0.7, 1.3, 1.7, 2.3, 2.7]
ax.bxp(stats1, showfliers=False, showmeans=True, positions=p)


# Simulations
# xy = [x[0], x[1], y[0], y[1], z[0], z[1], zz[0], zz[1]]
xy = [x[0], x[1], xx[0], xx[1], y[0], y[1], z[0], z[1]]
p1 = [3.5, 4.0, 4.9, 5.4, 6.4, 6.9, 7.9, 8.4]

ax.boxplot(xy, showfliers=False, showmeans=True, positions=p1)
ax.set_xticks([0.5, 1.5, 2.5, 3.75, 5.25, 6.65, 8.15])
ax.set_xticklabels(['MCC', 'HCC', 'IS', 'CL(1)', 'CL(2)','CL(3)', 'CL(4)'])

ax.set_ylabel("Average rally length")
ax.set_title("Game play performance over time")
ax.set_xlim((0,9))

line1 = ax.hlines(y=average_4, xmin=0, xmax=9, linestyles = '--', 
                  color='black', label = 'HCC level')
line2 = ax.hlines(y=average_6, xmin=0, xmax=9, linestyles = '--', 
                  color='red', label = 'Random (IS) level')
ax.legend(handles=[line1, line2])

fig.savefig('graph.png', dpi=500, bbox_inches='tight')

#%% T-tests

cl_t_test_2 = scipy.stats.ttest_ind(x[0], x[1])
cl_t_test_3 = scipy.stats.ttest_ind(y[0], y[1])
cl_t_test_4 = scipy.stats.ttest_ind(z[0], z[1])

print("t-test cl-2", cl_t_test_2)
print("t-test cl-3", cl_t_test_3)
print("t-test cl-4", cl_t_test_4)