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
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, plot_importance

# gpu card and data file
gpu = 'p100'
#gpu = 'titanx'
version = 'synthetic'
csv_file = "csvs/%s-%s-Performance.csv" % (gpu, version)

rng = np.random.RandomState(31337)

def mean_absolute_error(ground_truth, predictions):
    return np.mean(abs(ground_truth - predictions) / ground_truth)
    #return mean_squared_error(ground_truth, predictions)

def nn_fitting(X, y):

    nn_model = MLPRegressor(solver='adam', alpha=1e-4, activation='relu',
                    hidden_layer_sizes=(50, 20, 50), random_state=1, max_iter=1000)

    nn_model.fit(X, y)

    return nn_model

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

    ## random select test
    #for i in range(10):
    #    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    #    xgb_model = xgb.XGBRegressor().fit(X_train, y_train)
    #    predictions = xgb_model.predict(X_test)
    #    actuals = y_test
    #    #print mean_squared_error(actuals, predictions)
    #    print mean_absolute_error(actuals, predictions)

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    n_estimators = [50, 100, 150, 200]
    max_depth = [2, 4, 6, 8]
    learning_rate = [0.1, 0.01, 0.001, 0.0001]
    min_child_weight = [1, 2, 3, 4, 5]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, min_child_weight=min_child_weight)
    
    # xg_model = GridSearchCV(XGBRegressor(verbose=True), cv=10, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=True)
    xg_model = GridSearchCV(XGBRegressor(verbose=True), cv=10, param_grid=param_grid, scoring=loss, n_jobs=-1, verbose=True)
    xg_model.fit(X, y)
    # print xg_model.grid_scores_
    print xg_model.best_params_
    # print xg_model.best_score_

    #xg_model = xgb.XGBRegressor(max_depth=2, n_estimators=20, min_child_weight=1, learning_rate=0.1, verbose=True)
    #xg_model.fit(X, y)

    #print xg_model.feature_importances_
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
    svr_model = GridSearchCV(SVR(verbose=True, max_iter=1e6), cv=10, scoring='neg_mean_squared_error', param_grid=tuned_parameters)
    #svr_model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon, verbose=True, max_iter=-1)

    # Fit regression model
    svr_model.fit(X, y)

    # print svr_model.grid_scores_
    print svr_model.best_params_
    # print svr_model.best_score_

    return svr_model


def data_prepare(gpucard, version, csv_perf):

    if 'gtx980' in gpucard:
        GPUCONF = GTX980()
    elif 'p100' in gpucard:
        GPUCONF = P100()
    elif 'titanx' in gpucard:
        GPUCONF = TITANX()

    df = pd.read_csv(csv_perf, header = 0)
    
    #params = pd.DataFrame(columns=['n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'n_dm_ld', 'n_dm_st', 'n_flop_sp', 'mem_insts', 'insts']) 
    params = pd.DataFrame(columns=['n_gld', 'n_gst', 'gld_trans_per_req', 'gst_trans_per_req', \
				   'n_dm_ld', 'n_dm_st', \
				   'n_l2_ld', 'n_l2_st', \
				   'n_shm_ld', 'n_shm_st', 'shld_trans_per_req', 'shst_trans_per_req', \
				   'tex_hit_rate', 'tex_trans', \
				   'n_flop_sp', 'n_flop_sp_fma', 'n_flop_sp_spec', 'n_flop_dp', 'n_flop_dp_fma', \
				   ]) 
    
    # hardware parameters
    df['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    
    # global memory information
    params['n_gld'] = df['gld_transactions'] / df['warps'] 
    params['n_gst'] = df['gst_transactions'] / df['warps']
    params['gld_trans_per_req'] = df['gld_transactions_per_request'] 
    params['gst_trans_per_req'] = df['gst_transactions_per_request']
    params['n_gm'] = params['n_gld'] + params['n_gst']
    params['gm_req'] = params['n_gld'] / params['gld_trans_per_req'] + params['n_gst'] / params['gst_trans_per_req']
    
    # dram memory information
    params['n_dm_ld'] = df['dram_read_transactions'] / df['warps']
    params['n_dm_st'] = df['dram_write_transactions'] / df['warps']
    params['n_dm'] = params['n_dm_ld'] + params['n_dm_st'] 
    
    # l2 cache information
    params['n_l2_ld'] = df['l2_read_transactions'] / df['warps']
    params['n_l2_st'] = df['l2_write_transactions'] / df['warps']
    params['n_l2'] = params['n_l2_ld'] + params['n_l2_st']
    
    # shared memory information
    params['n_shm_ld'] = df['shared_load_transactions'] / df['warps'] 
    params['n_shm_st'] = df['shared_store_transactions'] / df['warps'] 
    params['shld_trans_per_req'] = df['shared_load_transactions_per_request']
    params['shst_trans_per_req'] = df['shared_store_transactions_per_request'] 
    params['n_shm'] = params['n_shm_ld'] + params['n_shm_st'] 
    params['shm_req'] = params['n_shm_ld'] / params['shld_trans_per_req'] + params['n_shm_st'] / params['shst_trans_per_req']
    
    # texture memory information
    params['tex_hit_rate'] = df['tex_cache_hit_rate']
    params['tex_trans'] = df['tex_cache_transactions'] / df['warps']

    # compute insts
    params['n_flop_sp'] = df['flop_count_sp'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    params['n_flop_sp_fma'] = df['flop_count_sp_fma'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    params['n_flop_sp_spec'] = df['flop_count_sp_special'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    params['n_flop_dp'] = df['flop_count_dp'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    params['n_flop_dp_fma'] = df['flop_count_dp_fma'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    params['n_int'] = df['inst_integer'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM

    # instruction statistic
    params['inst_per_warp'] = df['inst_per_warp']

    ## other parameters
    #df['mem_insts'] = params['n_gld'] + params['n_gst'] + params['n_shm_ld'] + params['n_shm_st']
    #params['other_insts'] = (df['inst_per_warp'] - df['mem_insts'] - params['n_flop_sp']) * 1.0 # / GPUCONF.CORES_SM
    #params.loc[params['other_insts'] < 0, 'other_insts'] = 0
    ## print params['other_insts']
    
    # grouth truth cycle per SM per round
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy']))
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT))
    #print params['real_cycle']
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / df['warps']
    
    # normalize
    #params = params.div(params.loc[:, params.columns != 'real_cycle'].sum(axis=1), axis=0)
    params = params.div(params['inst_per_warp'], axis=0)
    
    # grouth truth IPC
    try:
        params['real_cycle'] = df['ipc']
    except Exception as e:
        params['real_cycle'] = df['executed_ipc']

    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy'])) / params['inst_per_warp']
    #print params['real_cycle']
    
    # hardware information, frequency ratio, core/mem
    params['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    params['act_util'] = df['achieved_occupancy']
    
    # X = params.loc[:, params.columns != 'real_cycle']
    X = params.loc[:, ['n_dm', 'n_l2', 'n_shm', 'tex_hit_rate', 'tex_trans', 'n_flop_sp', 'n_flop_dp', 'act_util']]
    y = params['real_cycle']
    
    # X = X.div(X.loc[:, :].sum(axis=1), axis=0)

    print "Total number of samples:", len(X)
    X = X.astype(np.float64)
    print X.dtypes
    print X.head(5)

    params['appName'] = df['appName']
    features = params.loc[:, ['appName', 'c_to_m', 'n_gm', 'n_dm', 'n_l2', 'n_shm', 'tex_hit_rate', 'tex_trans', 'n_flop_sp', 'n_flop_dp', 'act_util', 'real_cycle']]
    features.to_csv("csvs/%s-%s-features.csv" % (gpucard, version))

    return X, y, df

def compare(train_X, train_y, test_X, test_y):
    print train_X.head(5)
    print test_X.head(5)
    print train_y[:5]
    print test_y[:5]

def train(train_X, train_y, train_df):

    print "len of train:", len(train_X), len(train_y), len(train_df)
    
    # fit train data and test on test data
    #fit_model = svr_fitting(gpu_X, gpu_y, 'rbf')
    #fit_model = rt_fitting(train_X, train_y)
    fit_model = xg_fitting(train_X, train_y)
    #fit_model = nn_fitting(train_X, train_y)

    train_y_pred = fit_model.predict(train_X)
    train_mae = mean_absolute_error(train_y, train_y_pred)
    
    print "Train Mean absolute error:", train_mae

    return fit_model

def test(model, test_X, test_y, test_df):
    test_y_pred = model.predict(test_X) 
    test_mae = mean_absolute_error(test_y, test_y_pred)
    
    ## fit all data/modeling
    #fit_model = svr_fitting(X, y, 'rbf')
    ##fit_model = rt_fitting(X, y)
    #pred_y = fit_model.predict(X)
    #mae = mean_absolute_error(y, pred_y)
    
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


gpu_X, gpu_y, gpu_df = data_prepare(gpu, version, csv_file)
test_X, test_y, test_df = data_prepare(gpu, 'real', './csvs/%s-real-Performance.csv' % gpu) 

# kernel_idx = range(0, len(gpu_X))
# random.shuffle(kernel_idx)
# train_len = len(gpu_X) * 9 / 10
# test_len = len(gpu_X) - train_len
# train_idx = kernel_idx[:train_len]
# test_idx = kernel_idx[train_len:]
# 
# print train_idx, test_idx
# 
# train_X = gpu_X.loc[train_idx, :]
# train_y = gpu_y[train_idx]
# train_df = gpu_df.loc[train_idx, :]
# test_X = gpu_X.loc[test_idx, :]
# test_y = gpu_y[test_idx]
# test_df = gpu_df.loc[test_idx, :]

model = train(gpu_X, gpu_y, gpu_df)
test(model, test_X, test_y, test_df)
