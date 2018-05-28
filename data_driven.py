import pandas as pd
import numpy as np
import sys
import random
from settings import *
from sklearn.svm import SVR
# from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV  
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

def mean_absolute_error(ground_truth, predictions):
    return np.mean(abs(ground_truth - predictions) / ground_truth)

def rt_fitting(X, y):

    # regr = DecisionTreeRegressor(max_depth=5)
    regr = RandomForestRegressor(max_depth=5, random_state=0)
    regr.fit(X, y)
    return regr

def svr_fitting(X, y, kernel, gamma=0.1, C=1e3, epsilon=0.2):

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=True)

    # initial svr model
    svr_model = GridSearchCV(SVR(kernel='rbf', verbose=True, max_iter=-1), cv=10, scoring=loss,
                           param_grid={"C": [1e3],  
                           "gamma": [0.1],
                           "epsilon": [0.2]})  
    #svr_model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon, verbose=True, max_iter=50000)

    # Fit regression model
    svr_model.fit(X, y)
    print svr_model.cv_results_
    print svr_model.best_params_

    return svr_model

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
# params.loc[params['other_insts'] < 0, 'other_insts'] = 0
# print params['other_insts']

# grouth truth cycle per SM per round
#params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (WARPS_MAX * SM_COUNT * df['achieved_occupancy']))
params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (WARPS_MAX * SM_COUNT))
#params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / df['warps']

# normalize
params = params.div(params.loc[:, params.columns != 'real_cycle'].sum(axis=1), axis=0)

## grouth truth IPC
#params['real_cycle'] = df['inst_per_warp'] * df['warps'] / (df['time/ms'] * df['coreF'] * 1000)

# frequency ratio, core/mem
params['c_to_m'] = df['coreF'] * 1.0 / df['memF']

# sm utilization
params['act_util'] = df['achieved_occupancy']

print params.head(10)

X = params.loc[:, params.columns != 'real_cycle']
y = params['real_cycle']

# split train/test dataset
#train_X, test_X, train_y, test_y = cross_validation.train_test_split(X, y ,test_size=0.2)
train_X, test_X, train_y, test_y = train_test_split(X, y ,test_size=0.1)

#train_num = len(X) * 9 / 10
#test_num = len(X) - train_num
#
#train_X = X[:train_num]
#train_y = y[:train_num]
#test_X = X[train_num:]
#test_y = y[train_num:]

fit_model = svr_fitting(train_X, train_y, 'rbf')
#fit_model = rt_fitting(train_X, train_y)
pred_y = fit_model.predict(test_X)

mae = mean_absolute_error(test_y, pred_y)

print "Mean absolute error:", mae

#for i in range(len(y)):
#    print i, y[i], y_rbf[i]

#kernels = df['appName'].drop_duplicates()
#for kernel in kernels:
#    tmp_y = y[df['appName'] == kernel]
#    tmp_pred_y = pred_y[df['appName'] == kernel]
#    
#    tmp_ape = np.mean(abs(tmp_y - tmp_pred_y) / tmp_y)
#    # if tmp_ape > 0.15:
#    print "%s:%f." % (kernel, tmp_ape)

