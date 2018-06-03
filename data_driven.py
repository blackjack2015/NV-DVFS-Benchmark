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

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    tuned_parameters = {'max_depth': [3, 4, 5, 6]}
    regr = RandomForestRegressor(random_state=0, verbose=True)
    # regr = DecisionTreeRegressor(max_depth=5)

    regr_model = GridSearchCV(regr, cv=5, scoring=loss, param_grid=tuned_parameters)
    regr_model.fit(X, y)
    return regr_model

def svr_fitting(X, y, kernel, gamma=1, C=1e4, epsilon=0.1):

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.5, 1], 'C': [1, 100, 10000], 'epsilon': [0.1, 0.2, 0.4]},
                        {'kernel': ['poly'], 'gamma': [0.1, 0.5, 1], 'C': [1, 10, 100, 1000], 'epsilon': [0.1, 0.2, 0.4], 'degree': [1, 2, 3]}]

    # initial svr model
    svr_model = GridSearchCV(SVR(verbose=True, max_iter=1e6), cv=10, scoring=loss, param_grid=tuned_parameters[1])
    #svr_model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon, verbose=True, max_iter=-1)

    # Fit regression model
    svr_model.fit(X, y)

    print svr_model.grid_scores_
    print svr_model.best_params_
    print svr_model.best_score_

    return svr_model


def data_prepare(gpucard, csv_perf):

    if gpucard == 'gtx980':
        GPUCONF = GTX980()
    elif gpucard == 'p100':
        GPUCONF = P100()
    elif gpucard == 'titanx':
        GPUCONF = TITANX()

    df = pd.read_csv(csv_perf, header = 0)
    
    #params = pd.DataFrame(columns=['n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'n_dm_ld', 'n_dm_st', 'n_flop_sp', 'mem_insts', 'insts']) 
    params = pd.DataFrame(columns=['n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'n_dm_ld', 'n_dm_st', 'n_flop_sp']) 
    
    # hardware parameters
    df['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    
    # shared memory information
    params['n_shm_ld'] = df['shared_load_transactions'] / df['warps'] 
    params['n_shm_st'] = df['shared_store_transactions'] / df['warps'] 
    
    # compute insts
    params['n_flop_sp'] = df['flop_count_sp'] / df['warps'] / GPUCONF.CORES_SM
    
    # global memory information
    params['n_gld'] = df['l2_read_transactions'] / df['warps']
    params['n_gst'] = df['l2_write_transactions'] / df['warps']
    
    # dram memory information
    params['n_dm_ld'] = df['dram_read_transactions'] / df['warps'] #/ (GPUCONF.WIDTH_MEM / 256)
    params['n_dm_st'] = df['dram_write_transactions'] / df['warps'] #/ (GPUCONF.WIDTH_MEM / 256)
    
    # other parameters
    df['mem_insts'] = params['n_gld'] + params['n_gst'] + params['n_shm_ld'] + params['n_shm_st']
    # params['other_insts'] = (df['inst_per_warp'] - df['mem_insts'] - params['n_flop_sp']) #/ GPUCONF.CORES_SM
    # params.loc[params['other_insts'] < 0, 'other_insts'] = 0
    # print params['other_insts']
    
    # grouth truth cycle per SM per round
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy']))
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT))
    #print params['real_cycle']
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / df['warps']
    
    # normalize
    params = params.div(params.loc[:, params.columns != 'real_cycle'].sum(axis=1), axis=0)
    
    # grouth truth IPC
    params['real_cycle'] = df['inst_per_warp'] * df['warps'] / (df['time/ms'] * df['coreF'] * 1000) / GPUCONF.SM_COUNT
    #print params['real_cycle']
    
    # frequency ratio, core/mem
    params['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    
    # sm utilization
    params['act_util'] = df['achieved_occupancy']
    
    print params.head(5)
    
    X = params.loc[:, params.columns != 'real_cycle']
    y = params['real_cycle']
    
    print "Total number of samples:", len(X)

    params['appName'] = df['appName']
    params.to_csv("csvs/%s_features.csv" % gpucard)

    return X, y, df

def compare(train_X, train_y, test_X, test_y):
    print train_X.head(5)
    print test_X.head(5)
    print train_y[:5]
    print test_y[:5]

# gpu card and data file
gpu1 = 'gtx980'
gpu2 = 'p100'
csv_temp = "csvs/%s-DVFS-Performance.csv"

# training data and test data are from different GPU cards
train_X, train_y, train_df = data_prepare(gpu1, csv_temp % gpu1)
test_X, test_y, test_df = data_prepare(gpu2, csv_temp % gpu2)

#compare(train_X, train_y, test_X, test_y)
#sys.exit(0)

# modeling accuracy, that just indicates the correlations between input features and target
#train_X, train_y, train_df = data_prepare(gpu1, csv_temp % gpu1)
#test_X, test_y, test_df = data_prepare(gpu1, csv_temp % gpu1)

# training data and test data are from the same GPU card
#train_X, test_X, train_y, test_y = train_test_split(X, y ,test_size=0.1)
#split_point = len(X) / 20 * 17
#train_X = X[:split_point]
#test_X = X[split_point:]
#train_y = y[:split_point]
#test_y = y[split_point:]

# fit train data and test on test data
# fit_model = svr_fitting(train_X, train_y, 'rbf')
fit_model = rt_fitting(train_X, train_y)
train_y_pred = fit_model.predict(train_X)
test_y_pred = fit_model.predict(test_X)
train_mae = mean_absolute_error(train_y, train_y_pred)
test_mae = mean_absolute_error(test_y, test_y_pred)

## fit all data/modeling
#fit_model = svr_fitting(X, y, 'rbf')
##fit_model = rt_fitting(X, y)
#pred_y = fit_model.predict(X)
#mae = mean_absolute_error(y, pred_y)

print "Train Mean absolute error:", train_mae
print "Test Mean absolute error:", test_mae

#for i in range(len(test_y)):
#    print i, test_y[i], pred_y[i]

kernels = test_df['appName'].drop_duplicates()
for kernel in kernels:
    tmp_y = test_y[test_df['appName'] == kernel]
    tmp_pred_y = test_y_pred[test_df['appName'] == kernel]
    
    tmp_ape = np.mean(abs(tmp_y - tmp_pred_y) / tmp_y)
    # if tmp_ape > 0.15:
    print "%s:%f." % (kernel, tmp_ape)

