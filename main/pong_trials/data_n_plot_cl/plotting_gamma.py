#!/usr/bin/env python
# coding: utf-8

#%% Import all modules. Set path for importing dishpill_models

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.max_columns = 150

sns.set()
pd.set_option('display.max_rows', 250)
sns.set_style("whitegrid")

import warnings
warnings.filterwarnings('ignore')
path = "../models"
if not path in sys.path:
    sys.path.append(path)

df99 = pd.DataFrame()

new_data = pd.read_csv('gamma_cl_M1.csv')
new_data['group'] = 0
new_data['group_name'] = 'CL-1'
df99 = pd.concat([df99, new_data])

new_data = pd.read_csv('gamma_cl_M2.csv')
new_data['group'] = 1
new_data['group_name'] = 'CL-2'
df99 = pd.concat([df99, new_data])

new_data = pd.read_csv('gamma_cl_M3.csv')
new_data['group'] = 2
new_data['group_name'] = 'CL-3'
df99 = pd.concat([df99, new_data])

new_data = pd.read_csv('gamma_cl_M4.csv')
new_data['group'] = 3
new_data['group_name'] = 'CL-4'
df99 = pd.concat([df99, new_data])

#Converting Gamma vec to log-scale to amplify difference
df99['gamma'] += 1e-15
df99['gamma'] = np.log(df99['gamma'])

#%%Gamma box-plot
plt.clf()

df_test2 = df99

labels = df_test2.group_name.unique()
x_pos = np.arange(len(labels))

x = df_test2['group_name']
y = df_test2['gamma']

hue = df_test2['half']

sns.set(style="darkgrid")
sns.set(font_scale=1.4)

ax = sns.boxplot(data=df99, x=x, y=y, hue=hue, palette="Set2", showfliers=False,
                 showmeans = True,
                 meanprops= {"markerfacecolor":"black",
                       "markeredgecolor":"black",
                      "markersize":"5"})

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=16)
ax.set_ylabel('Average Risk (Gamma) in log-scale',fontsize = 12)
ax.set_xlabel('Group',fontsize = 18)
ax.grid(False)
ax.legend([0, 1], ["0-5", "6-20"], fontsize = 14)

L = plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1),
               title = "Minutes", borderaxespad=0.1, frameon=False)

L.get_texts()[0].set_text('0-5')
L.get_texts()[1].set_text('6-20')

plt.savefig('gamma_CL_boxplot.png',dpi = 500, bbox_inches='tight')
plt.show()

#%% Gamma Reg plot

plt.clf()

df2 = df99.groupby(['group',
                   'session_num',
                   'group_name',
                   'elapse_minute_rounded']).mean(numeric_only = True)

lines = df2.reset_index()

sns.set(font_scale=1.4)
f, ax = plt.subplots(figsize=(7,7))

for i in lines.group.unique():
    
    control = lines[lines['group'] == i]
    label = control.group_name.unique()[0]
    
    px = control[control['elapse_minute_rounded']<21]['elapse_minute_rounded']
    py = control[control['elapse_minute_rounded']<21]['gamma']
    
    sns.regplot(x=ax.xaxis.convert_units(px), y=py, x_estimator=np.mean, ci=95, 
            scatter = False, order = 1, label = label, fit_reg=True)

sns.set(style="darkgrid")

ax.set_ylabel('Average Risk (Gamma) in log-scale',fontsize =20)
ax.set_xlabel('Elapsed Minute',fontsize =20)
plt.legend(loc='upper left',fontsize =14)

plt.savefig('gamma_regression.png',dpi = 500, bbox_inches='tight')
plt.show()
