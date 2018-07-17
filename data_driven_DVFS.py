import pandas as pd
import numpy as np
import sys
import random
from settings import *
from sklearn.svm import SVR
# from sklearn import cross_validation
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, plot_importance

rng = np.random.RandomState(31337)

def mean_absolute_error(ground_truth, predictions):
    return np.mean(abs(ground_truth - predictions) / ground_truth)
    #return mean_squared_error(ground_truth, predictions)

def xg_fitting(X, y):

    #split_point = 885
    #xgb_model = xgb.XGBRegressor().fit(X[:split_point], y[:split_point])
    #predictions = xgb_model.predict(X[split_point:])
    #actuals = y[split_point:]
    #print mean_squared_error(actuals, predictions)
    #print mean_absolute_error(actuals, predictions)

    #kf = KFold(n_splits=10, shuffle=True, random_state=rng)
    #for train_index, test_index in kf.split(X):
    #    xgb_model = xgb.XGBRegressor().fit(X.loc[train_index], y[train_index])
    #    predictions = xgb_model.predict(X.loc[test_index])
    #    actuals = y[test_index]
    #    #print mean_squared_error(actuals, predictions)
    #    print mean_absolute_error(actuals, predictions)

    # random select test
    for i in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
        xgb_model = xgb.XGBRegressor().fit(X_train, y_train)
        predictions = xgb_model.predict(X_test)
        actuals = y_test
        #print mean_squared_error(actuals, predictions)
        print mean_absolute_error(actuals, predictions)

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    n_estimators = [50, 100, 150, 200]
    max_depth = [2, 4, 6, 8]
    learning_rate = [0.1, 0.01, 0.001]
    min_child_weight = [1, 2, 3, 4, 5]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, min_child_weight=min_child_weight)
    
    #xg_model = GridSearchCV(XGBRegressor(verbose=True), cv=10, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=True)
    #xg_model.fit(X, y)
    #print xg_model.grid_scores_
    #print xg_model.best_params_
    #print xg_model.best_score_

    xg_model = xgb.XGBRegressor(max_depth=4, n_estimators=100, min_child_weight=5, learning_rate=0.1, verbose=True)
    xg_model.fit(X, y)

    print xg_model.feature_importances_
    #plot_importance(xg_model)
    #plt.show()

    return xg_model

def rt_fitting(X, y):

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    tuned_parameters = {'max_depth': [3, 4, 5, 6]}
    regr = RandomForestRegressor(random_state=0, verbose=True)
    # regr = DecisionTreeRegressor(max_depth=5)

    regr_model = GridSearchCV(regr, cv=10, scoring='neg_mean_squared_error', n_jobs=-1, param_grid=tuned_parameters)
    regr_model.fit(X, y)

    print regr_model.grid_scores_
    print regr_model.best_params_
    print regr_model.best_score_

    return regr_model

def svr_fitting(X, y, kernel, gamma=1, C=1e4, epsilon=0.1):

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.1, 0.5, 1], 'C': [1, 100, 10000], 'epsilon': [0.1, 0.2, 0.4]},
                        {'kernel': ['poly'], 'gamma': [0.1, 0.5, 1], 'C': [1, 10, 100, 1000], 'epsilon': [0.1, 0.2, 0.4], 'degree': [1, 2, 3]}]

    # initial svr model
    svr_model = GridSearchCV(SVR(verbose=True, max_iter=1e6), cv=10, scoring='neg_mean_squared_error', param_grid=tuned_parameters[0])
    #svr_model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon, verbose=True, max_iter=-1)

    # Fit regression model
    svr_model.fit(X, y)

    print svr_model.grid_scores_
    print svr_model.best_params_
    print svr_model.best_score_

    return svr_model


def data_prepare(gpucard, csv_perf):

    if 'gtx980' in gpucard:
        GPUCONF = GTX980()
    elif 'p100' in gpucard:
        GPUCONF = P100()
    elif 'titanx' in gpucard:
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
    params['n_flop_sp'] = df['flop_count_sp'] * 1.0 / df['warps'] # / GPUCONF.CORES_SM
    
    # global memory information
    params['n_gld'] = df['l2_read_transactions'] / df['warps'] 
    params['n_gst'] = df['l2_write_transactions'] / df['warps']
    
    # dram memory information
    params['n_dm_ld'] = df['dram_read_transactions'] / df['warps']
    params['n_dm_st'] = df['dram_write_transactions'] / df['warps']
    
    # other parameters
    df['mem_insts'] = params['n_gld'] + params['n_gst'] + params['n_shm_ld'] + params['n_shm_st']
    params['other_insts'] = (df['inst_per_warp'] - df['mem_insts'] - params['n_flop_sp']) * 1.0 # / GPUCONF.CORES_SM
    params.loc[params['other_insts'] < 0, 'other_insts'] = 0
    # print params['other_insts']
    
    # grouth truth cycle per SM per round
    params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy']))
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT))
    #print params['real_cycle']
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / df['warps']
    
    # normalize
    params = params.div(params.loc[:, params.columns != 'real_cycle'].sum(axis=1), axis=0)
    
    # grouth truth IPC
    #params['real_cycle'] = df['ipc']
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
gpu = 'titanx'
version = 'single-workload'
csv_file = "csvs/%s-%s-DVFS-Performance.csv" % (gpu, version)

# pre-load gpu data
gpu_X, gpu_y, gpu_df = data_prepare(gpu, csv_file)
c_to_m_set = list(gpu_df['c_to_m'].drop_duplicates())
print c_to_m_set

random.shuffle(c_to_m_set)
train_c_to_m = c_to_m_set[:len(c_to_m_set) * 10 / 16]
print train_c_to_m

train_X = gpu_X[gpu_X['c_to_m'].isin(train_c_to_m)]
train_y = gpu_y[gpu_X['c_to_m'].isin(train_c_to_m)]
train_df = gpu_df[gpu_X['c_to_m'].isin(train_c_to_m)]
test_X = gpu_X[~gpu_X['c_to_m'].isin(train_c_to_m)]
test_y = gpu_y[~gpu_X['c_to_m'].isin(train_c_to_m)]
test_df = gpu_df[~gpu_X['c_to_m'].isin(train_c_to_m)]

print "len of train:", len(train_X), train_X.tail(3)
print "len of test:", len(test_X), test_X.tail(3)

# fit train data and test on test data
#fit_model = svr_fitting(train_X, train_y, 'rbf')
#fit_model = rt_fitting(train_X, train_y)
fit_model = xg_fitting(train_X, train_y)
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

