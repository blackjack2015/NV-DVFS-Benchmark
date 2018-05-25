import pandas as pd
import numpy as np
import sys
from settings import *
from sklearn.svm import SVR
import matplotlib.pyplot as plt

df = pd.read_csv(csv_perf, header = 0)

#params = pd.DataFrame(columns=['n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'n_dm_ld', 'n_dm_st', 'n_flop_sp', 'mem_insts', 'insts']) 
params = pd.DataFrame(columns=['n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'n_dm_ld', 'n_dm_st', 'n_flop_sp']) 

# shared memory information
params['n_shm_ld'] = df['shared_load_transactions'] / df['warps']
params['n_shm_st'] = df['shared_store_transactions'] / df['warps']

# global memory information
params['n_gld'] = df['l2_read_transactions'] / df['warps']
params['n_gst'] = df['l2_write_transactions'] / df['warps']

# dram memory information
params['n_dm_ld'] = df['dram_read_transactions'] / df['warps']
params['n_dm_st'] = df['dram_write_transactions'] / df['warps']

# compute insts
params['n_flop_sp'] = df['flop_count_sp'] / df['warps']

# other parameters
df['mem_insts'] = params['n_gld'] + params['n_gst'] + params['n_shm_ld'] + params['n_shm_st']
params['other_insts'] = df['inst_per_warp'] - df['mem_insts'] - params['n_flop_sp']

# grouth truth cycle per SM per round
#params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (WARPS_MAX * SM_COUNT * df['achieved_occupancy']))
params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (WARPS_MAX * SM_COUNT))
#params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / df['warps']

# normalize
params = params.div(params.loc[:, params.columns != 'real_cycle'].sum(axis=1), axis=0)

# frequency ratio, core/mem
#params['c_to_m'] = df['coreF'] * 1.0 / df['memF']

# sm utilization
#params['act_util'] = df['achieved_occupancy']

print params.head(10)

X = params.loc[:, params.columns != 'real_cycle']
y = params['real_cycle']

# Fit regression model
svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
svr_lin = SVR(kernel='linear', C=1e3)
svr_poly = SVR(kernel='poly', C=1e3, degree=1)
y_rbf = svr_rbf.fit(X, y).predict(X)
y_lin = svr_lin.fit(X, y).predict(X)
y_poly = svr_poly.fit(X, y).predict(X)

ape_y_rbf = np.mean(abs(y - y_rbf) / y)
ape_y_lin = np.mean(abs(y - y_lin) / y)
ape_y_poly = np.mean(abs(y - y_poly) / y)

print ape_y_rbf, ape_y_lin, ape_y_poly

#for i in range(len(y)):
#    print i, y[i], y_rbf[i]

kernels = df['appName'].drop_duplicates()
for kernel in kernels:
    tmp_y = y[df['appName'] == kernel]
    tmp_y_rbf = y_rbf[df['appName'] == kernel]
    
    tmp_ape = np.mean(abs(tmp_y - tmp_y_rbf) / tmp_y)
    # if tmp_ape > 0.15:
    print "%s:%f." % (kernel, tmp_ape)

