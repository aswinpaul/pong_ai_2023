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

cl_data_1 = pd.read_csv('gamma_cl_M1.csv')
cl_data_1['group'] = 0
cl_data_1['group_name'] = 'CL (1)'

cl_data_2 = pd.read_csv('gamma_cl_M2.csv')
cl_data_2['group'] = 1
cl_data_2['group_name'] = 'CL (2)'

cl_data_3 = pd.read_csv('gamma_cl_M3.csv')
cl_data_3['group'] = 2
cl_data_3['group_name'] = 'CL (3)'

cl_data_4 = pd.read_csv('gamma_cl_M4.csv')
cl_data_4['group'] = 3
cl_data_4['group_name'] = 'CL (4)'

df4 = pd.concat([cl_data_1, cl_data_2, cl_data_3, cl_data_4])

#%%Gamma with time

df_test2 = df4

labels = df_test2.group_name.unique()
x_pos = np.arange(len(labels))

x = df_test2['group_name']
y = df_test2['gamma_min']

hue = df_test2['half']

sns.set(style="darkgrid")
sns.set(font_scale=1.4)

ax = sns.boxplot(data=df4, x=x, y=y, hue=hue, palette="Set2", showfliers=False,
                 showmeans = True,
                 meanprops={"markerfacecolor":"black",
                       "markeredgecolor":"black",
                      "markersize":"5"})
ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=16)
ax.axhline(0.5, color='red',ls=':')
# ax.set_yticklabels([0,0,5,10,15,20,25,30], fontsize=16)
# ax.set_title('Pong Performance over Time With All Features')

ax.set_ylabel('Gamma (min)',fontsize = 18)
ax.set_xlabel('Group',fontsize = 18)
ax.grid(False)
ax.legend([0, 1], ["0-5", "6-20"], fontsize = 14)

L = plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1),
               title = "Minutes", borderaxespad=0.1, frameon=False)

L.get_texts()[0].set_text('0-5')
L.get_texts()[1].set_text('6-20')

plt.savefig('gamma_CL.png', bbox_inches='tight')

#%% Reg plot long rally

df2 = df4.groupby(['group',
                   'session_num', 
                   'elapse_minute_rounded']).mean(numeric_only = True)

cleanDF = df2

lines = cleanDF.reset_index()

control = lines[(lines.group == 0)]
control['Zhit_count'] = (control.gamma_min - 
                         control.gamma_min.mean())/control.gamma_min.std(ddof=0)
control['Zhit_count'] = control['Zhit_count'].abs()
control = control[control.Zhit_count <= 2]

human = lines[(lines.group == 1)]
human['Zhit_count'] = (human.gamma_min - 
                       human.gamma_min.mean())/human.gamma_min.std(ddof=0)
human['Zhit_count'] = human['Zhit_count'].abs()
human = human[human.Zhit_count <= 2]

cl = lines[(lines.group == 2)]
cl['Zhit_count'] = (cl.gamma_min - 
                     cl.gamma_min.mean()) / cl.gamma_min.std(ddof=0)
cl['Zhit_count'] = cl['Zhit_count'].abs()
cl = cl[cl.Zhit_count <= 2]

cl2 = lines[(lines.group == 3)]
cl2['Zhit_count'] = (cl2.gamma_min - 
                     cl2.gamma_min.mean()) / cl2.gamma_min.std(ddof=0)
cl2['Zhit_count'] = cl2['Zhit_count'].abs()
cl2 = cl2[cl2.Zhit_count <= 2]

px = control[control['elapse_minute_rounded']<20]['elapse_minute_rounded']
py = control[control['elapse_minute_rounded']<20]['gamma_min']

hx = human[human['elapse_minute_rounded']<20]['elapse_minute_rounded']
hy = human[human['elapse_minute_rounded']<20]['gamma_min']

clx = cl[cl['elapse_minute_rounded']<20]['elapse_minute_rounded']
cly = cl[cl['elapse_minute_rounded']<20]['gamma_min']

cl2x = cl2[cl2['elapse_minute_rounded']<20]['elapse_minute_rounded']
cl2y = cl2[cl2['elapse_minute_rounded']<20]['gamma_min']

sns.set(font_scale=1.4)
f, ax = plt.subplots(figsize=(7,7))
sns.regplot(x=ax.xaxis.convert_units(px), y=py, x_estimator=np.mean, ci=95, scatter = False, order = 1, label = "CL (T = 1)",color='b')
sns.regplot(x=ax.xaxis.convert_units(hx), y=hy, x_estimator=np.mean, ci=95, scatter = False, order = 1, label = "CL (T = 2)",color = "#FF7D40")
sns.regplot(x=ax.xaxis.convert_units(clx), y=cly, x_estimator=np.mean, ci=95, scatter = False, order = 1, label = "CL (T = 3)",color = "g")
sns.regplot(x=ax.xaxis.convert_units(cl2x), y=cl2y, x_estimator=np.mean, ci=95, scatter = False, order = 1, label = "CL (T = 4)",color = "r")


sns.set(style="darkgrid")

ax.set_ylabel('Gamma (CL method)',fontsize =20)
ax.set_xlabel('Elapsed Minute',fontsize =20)
plt.xlim([-0.5, 19.5])
plt.legend(loc='upper left',fontsize =14)
ax.grid(False)

plt.savefig('gamma_CL_reg.png', bbox_inches='tight')