import pandas as pd
import numpy as np
import sys, argparse
import random
from settings import *
from sklearn.svm import SVR
# from sklearn import cross_validation
from sklearn.preprocessing import StandardScaler, MinMaxScaler
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


rng = np.random.RandomState(31337)
DATAROOT = "csvs/raw"
#out_kernels = ['binomialOptions', 'eigenvalues', 'scanUniformUpdate', 'stereoDisparity', 'reduction', 'matrixMulGlobal', 'cfd', 'hotspot', 'dxtc', 'backpropBackward']
#out_kernels = ['binomialOptions', 'eigenvalues', 'stereoDisparity', 'reduction', 'matrixMulGlobal', 'quasirandomGenerator', 'convolutionTexture']
#out_kernels = ['binomialOptions', 'eigenvalues', 'stereoDisparity', 'reduction', 'matrixMulGlobal', 'quasirandomGenerator']
out_kernels = []
#out_kernels = ['eigenvalues', 'matrixMulGlobal']

def mean_absolute_error(ground_truth, predictions):
    return np.mean(abs(ground_truth - predictions) / ground_truth)
    #return mean_squared_error(ground_truth, predictions)

def nn_fitting(X, y):

    # make score function
    #scaler = None
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    hidden_layer_sizes = [(20, 15, 10, 15, 20)]
    alpha = [1e-1, 1e-2, 1e-3, 1e-4]
    activation = ['relu', 'logistic', 'tanh']
    param_grid = dict(hidden_layer_sizes = hidden_layer_sizes, alpha = alpha, activation = activation)

    nn_model = MLPRegressor(solver='lbfgs', random_state=1, max_iter=60000, warm_start=True)

    nn_model = GridSearchCV(nn_model, cv=10, param_grid = param_grid, scoring='neg_mean_squared_error', n_jobs=8, verbose=True)
    nn_model.fit(X, y)

    print nn_model.best_params_

    #nn_model = MLPRegressor(solver='adam', hidden_layer_sizes = (10, 15, 10), alpha = 1e-5, random_state=1, max_iter=30000, warm_start=True)
    #nn_model.fit(X, y)
    return nn_model, scaler

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

    n_estimators = [300, 400, 500, 1000]
    max_depth = [3, 4, 5, 6]
    learning_rate = [0.3, 0.2, 0,1, 0.05]
    min_child_weight = [0.1, 0.5, 1, 2]
    #n_estimators = [1000]
    #max_depth = [4]
    #learning_rate = [0.1]
    #min_child_weight = [0.5]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, min_child_weight=min_child_weight)
    
    xg_model = GridSearchCV(XGBRegressor(verbose=False), cv=3, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=False)
    #xg_model = GridSearchCV(XGBRegressor(verbose=True, early_stopping_rounds=5), cv=10, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=True)
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

def svr_fitting(X, y, kernel = 'rbf'):

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.8, 1, 1.2], 'C': [10, 100, 1000], 'epsilon': [0.1, 0.4, 0.8]},
                        {'kernel': ['poly'], 'gamma': [0.5, 1, 2], 'C': [10, 100, 1000], 'epsilon': [0.1, 0.5, 1, 2], 'degree': [2, 3, 4, 5]}]

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.2], 'C': [10], 'epsilon': [0.1]},
                        {'kernel': ['poly'], 'gamma': [0.5], 'C': [10], 'epsilon': [0.1], 'degree': [2]}]

    kernel_idx = 0
    # initial svr model
    if kernel == 'rbf':
        kernel_idx = 0
    else:
        kernel_idx = 1

    svr_model = GridSearchCV(SVR(verbose=False, max_iter=1e6), cv=2, scoring='neg_mean_squared_error', param_grid=tuned_parameters[kernel_idx])
    #svr_model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon, verbose=True, max_iter=-1)

    # Fit regression model
    svr_model.fit(X, y)

    # print svr_model.grid_scores_
    print svr_model.best_params_
    # print svr_model.best_score_

    return svr_model

def train(X, y, ml_algo):

    #print "len of train:", len(train_X), len(train_y), len(train_df)
    
    # fit train data and test on test data
    scaler = None
    if ml_algo == 'svr-poly':
        fit_model = svr_fitting(X, y, 'poly')
    if ml_algo == 'svr-rbf':
        fit_model = svr_fitting(X, y, 'rbf')
    if ml_algo == 'xgboost':
        fit_model = xg_fitting(X, y)
    if ml_algo == 'nn':
        fit_model, scaler = nn_fitting(X, y)

    return fit_model, scaler

def test(model, X, y, scaler = None):
    if scaler != None:
        X = scaler.transform(X)
    y_pred = model.predict(X) 
    test_mae = mean_absolute_error(y, y_pred)
    
    y = list(y)
    y_pred = list(y_pred)
    #for i in range(len(y)):
    #    print i, y[i], y_pred[i]
    print "Test Mean absolute error:", test_mae
    return y_pred
    
    
    #kernels = test_df['appName'].drop_duplicates()

    #bias_level = [[], [], [], []]
    #for kernel in kernels:
    #    tmp_y = test_y[test_df['appName'] == kernel]
    #    tmp_pred_y = test_y_pred[test_df['appName'] == kernel]
    #    
    #    tmp_ape = np.mean(abs(tmp_y - tmp_pred_y) / tmp_y)
    #    if tmp_ape < 0.10:
    #        bias_level[0].append(tmp_ape)
    #    elif tmp_ape < 0.15:
    #        bias_level[1].append(tmp_ape)
    #    elif tmp_ape < 0.2:
    #        bias_level[2].append(tmp_ape)
    #    else:
    #        bias_level[3].append(tmp_ape)
    #    print "%s: %f" % (kernel, tmp_ape)

    #if len(bias_level[0]) != 0:
    #    print "Average error of    < 10% :", len(bias_level[0]), np.mean(bias_level[0])
    #if len(bias_level[1]) != 0:
    #    print "Average error of 10 ~ 15% :", len(bias_level[1]), np.mean(bias_level[1])
    #if len(bias_level[2]) != 0:
    #    print "Average error of 15 ~ 20% :", len(bias_level[2]), np.mean(bias_level[2])
    #if len(bias_level[3]) != 0:
    #    print "Average error of    > 20% :", len(bias_level[3]), np.mean(bias_level[3])

class GPU_Power:
    def __init__(self, gpucard, bench_conf, kernel_conf, algo):
        self.gpucard = gpucard
        self.bench_conf = bench_conf
        self.kernel_conf = kernel_conf
        self.method = algo

        if 'gtx980' in gpucard:
            if 'low-dvfs' in bench_conf:
                self.GPUCONF = GTX980(dvfs_range = 'low')
            else:
                self.GPUCONF = GTX980(dvfs_range = 'high')
        elif 'p100' in gpucard:
            self.GPUCONF = P100()
        elif 'titanx' in gpucard:
            self.GPUCONF = TITANX()
        elif 'gtx1080ti' in gpucard:
            self.GPUCONF = GTX1080TI()

        self.gpudata = 0
        self.X = 0
        self.y = 0
        self.data_prepare()
        self.get_kernel_set()

    def data_prepare(self):
    
        df = pd.read_csv("%s/%s-%s-Performance-Power.csv" % (DATAROOT, self.bench_conf, self.kernel_conf), header = 0)
        
        df = df[~df.appName.isin(out_kernels)]
        df = df.reset_index(drop=True)
    
        self.gpudata = df
        #params = pd.DataFrame(columns=['n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'n_dm_ld', 'n_dm_st', 'n_flop_sp', 'mem_insts', 'insts']) 
        params = pd.DataFrame(columns=['n_gld', 'n_gst', 'gld_trans_per_req', 'gst_trans_per_req', \
    				   'n_dm_ld', 'n_dm_st', \
    				   'n_l2_ld', 'n_l2_st', \
    				   'n_shm_ld', 'n_shm_st', 'shld_trans_per_req', 'shst_trans_per_req', \
    				   'tex_hit_rate', 'tex_trans', \
    				   'n_flop_sp', 'n_flop_sp_fma', 'n_flop_sp_spec', 'n_flop_dp', 'n_flop_dp_fma', 'n_int', \
    				   ]) 
        

        # global memory information
        params['n_gld'] = df['gld_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
        params['n_gst'] = df['gst_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
        params['gld_trans_per_req'] = df['gld_transactions_per_request'] 
        params['gst_trans_per_req'] = df['gst_transactions_per_request']
        params['gld_req'] = 0
        params.loc[params['gld_trans_per_req'] > 0, 'gld_req'] = params.loc[params['gld_trans_per_req'] > 0, 'n_gld'] / params.loc[params['gld_trans_per_req'] > 0, 'gld_trans_per_req']
        params['gst_req'] = 0
        params.loc[params['gst_trans_per_req'] > 0, 'gst_req'] = params.loc[params['gst_trans_per_req'] > 0, 'n_gst'] / params.loc[params['gst_trans_per_req'] > 0, 'gst_trans_per_req']
        params['n_gm'] = params['n_gld'] + params['n_gst']
        params['gm_req'] = params['gld_req'] + params['gst_req']
        
        # dram memory information
        params['n_dm_ld'] = df['dram_read_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
        params['n_dm_st'] = df['dram_write_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
        params['n_dm'] = params['n_dm_ld'] + params['n_dm_st'] 
        
        # l2 cache information
        params['n_l2_ld'] = df['l2_read_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
        params['n_l2_st'] = df['l2_write_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
        params['n_l2'] = params['n_l2_ld'] + params['n_l2_st']
        
        # shared memory information
        params['n_shm_ld'] = df['shared_load_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
        params['n_shm_st'] = df['shared_store_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
        params['shld_trans_per_req'] = df['shared_load_transactions_per_request']
        params['shst_trans_per_req'] = df['shared_store_transactions_per_request'] 
        params['shld_req'] = 0
        params.loc[params['shld_trans_per_req'] > 0, 'shld_req'] = params.loc[params['shld_trans_per_req'] > 0, 'n_shm_ld'] / params.loc[params['shld_trans_per_req'] > 0, 'shld_trans_per_req']
        params['shst_req'] = 0
        params.loc[params['shst_trans_per_req'] > 0, 'shst_req'] = params.loc[params['shst_trans_per_req'] > 0, 'n_shm_st'] / params.loc[params['shst_trans_per_req'] > 0, 'shst_trans_per_req']
        params['n_shm'] = params['n_shm_ld'] + params['n_shm_st'] 
        params['shm_req'] = params['shst_req'] + params['shld_req']
        
        # texture memory information
        params['tex_hit_rate'] = df['tex_cache_hit_rate']
        params['tex_trans'] = df['tex_cache_transactions'] / df['warps'] # / GPUCONF.LS_UNITS
    
        # compute insts
        params['n_flop_sp'] = df['flop_count_sp'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
        params['n_flop_sp_fma'] = df['flop_count_sp_fma'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
        params['n_flop_sp_spec'] = df['flop_count_sp_special'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
        #params['n_flop_sp'] -= params['n_flop_sp_spec']
        params['n_flop_dp'] = df['flop_count_dp'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
        params['n_flop_dp_fma'] = df['flop_count_dp_fma'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
        params['n_int'] = df['inst_integer'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    
        # branch
        params['branch'] = df['cf_executed'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    
        # instruction statistic
        params['inst_per_warp'] = df['inst_per_warp']
    
        ## other parameters
        #df['mem_insts'] = params['n_gld'] + params['n_gst'] + params['n_shm_ld'] + params['n_shm_st']
        #params['other_insts'] = (df['inst_per_warp'] - df['mem_insts'] - params['n_flop_sp']) * 1.0 # / GPUCONF.CORES_SM
        #params.loc[params['other_insts'] < 0, 'other_insts'] = 0
        ## print params['other_insts']
        
        params['real_cycle'] = None
        #params['avg_power'] = df['power/W']
        params['avg_power'] = None
        for idx, item in df.iterrows():
            params.loc[idx, 'real_cycle'] = float(item['time/ms'] * 1.0 / df[(df['appName'] == item['appName']) & (df['coreF'] == self.GPUCONF.CORE_FREQ) & (df['memF'] == self.GPUCONF.MEM_FREQ)]['time/ms'])
            params.loc[idx, 'avg_power'] = float(item['power/W'] * 1.0 / df[(df['appName'] == item['appName']) & (df['coreF'] == self.GPUCONF.CORE_FREQ) & (df['memF'] == self.GPUCONF.MEM_FREQ)]['power/W'])
    
        # grouth truth cycle per SM per round / ground truth IPC
        #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy']))
        #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT))
        #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / df['warps']
        #try:
        #    params['real_cycle'] = df['ipc']
        #except Exception as e:
        #    params['real_cycle'] = df['executed_ipc']
    
        # hardware information, frequency ratio, core/mem, runtime efficiency
        params['coreF'] = df['coreF'] * 1.0 / self.GPUCONF.CORE_FREQ
        params['memF'] = df['memF'] * 1.0 / self.GPUCONF.MEM_FREQ
        params['c_to_m'] = df['coreF'] * 1.0 / df['memF']
        params['act_util'] = df['achieved_occupancy']
        params['warp_eff'] = df['warp_execution_efficiency']
        try:
            params['sm_act'] = df['sm_activity']
        except Exception as e:
            params['sm_act'] = df['sm_efficiency']
        
        # select features for training
        #inst_features = ['n_dm', 'n_l2', 'n_shm', 'tex_trans', 'n_flop_sp', 'n_flop_dp', 'n_int', 'branch']
        #inst_features = ['n_dm', 'n_l2', 'n_shm', 'tex_trans', 'n_flop_sp', 'n_flop_dp', 'n_int', 'n_flop_sp_spec']
        inst_features = ['n_dm', 'n_l2', 'n_shm', 'tex_trans', 'n_flop_sp', 'n_flop_dp', 'n_int', 'branch']
    
        ## normalized with inst_per_warp, predict cycle per round
        #X = X.div(params['inst_per_warp'], axis=0)
        X = params.loc[:, inst_features]
        X = X.div(X.loc[:, :].sum(axis=1), axis=0)
        for feature in inst_features:
            X[feature] = X[feature] * params['act_util']
            X[feature] = X[feature] * params['sm_act']
            X[feature] = X[feature] * params['warp_eff']
        #y = params['real_cycle'] / params['inst_per_warp']
    
        # normalized with total amount of insts, predict cycle per round
        #y = params['real_cycle'].div(X.loc[:, :].sum(axis=1), axis=0) # for real cycle
        #y = params['real_cycle']
        y = params['avg_power']
    
        ## normalized with inst_per_warp, predict ipc
        #X = params.loc[:, inst_features]
        #X = X.div(params['inst_per_warp'], axis=0)
        #y = params['real_cycle']
    
        ## normalized with total amount of insts, predict ipc
        #X = params.loc[:, inst_features]
        #X = X.div(X.loc[:, :].sum(axis=1), axis=0)
        #y = params['real_cycle']
    
        util_features = ['act_util', 'warp_eff', 'sm_act', 'coreF', 'memF']
        for uf in util_features:
            X[uf] = params[uf]
        
        
        #print "Total number of samples:", len(X)
        X = X.astype(np.float64)
        #print X.head(5)
        #print y.head(5)
    
        features = pd.DataFrame([])
        features['appName'] = df['appName']
        for f in X.keys():
            features[f] = X[f]
        #features['inst_per_warp'] = params['inst_per_warp']
        features['avg_power'] = y
        features.to_csv("csvs/%s-%s-features.csv" % (self.bench_conf, self.kernel_conf))
    
        #return X, y, df
        self.X = X
        self.y = y
        print "total sample number:", len(X)

    def get_kernel_set(self):
        self.kernel_set = list(self.gpudata['appName'].drop_duplicates())
        print "kernel number:", len(self.kernel_set)

    def get_freq_set(self):
        self.core_freq_set = list(self.gpudata['coreF'].drop_duplicates())
        self.mem_freq_set = list(self.gpudata['memF'].drop_duplicates())

    def split_data(self, mode = 'random', test_size = 0.3):
        if mode == 'random':
            self.train_X, self.test_X, self.train_y, self.test_y = train_test_split(self.X, self.y, test_size=0.3)
        elif mode == 'kernel':
            random.shuffle(self.kernel_set)
            train_kernel = self.kernel_set[:int(len(self.kernel_set) * (1 - 0.3))]
            test_kernel = self.kernel_set[int(len(self.kernel_set) * (1 - 0.3)):]
            train_idx = self.gpudata['appName'].isin(train_kernel)
            test_idx = self.gpudata['appName'].isin(test_kernel)

            self.train_X = self.X[train_idx]
            self.train_y = self.y[train_idx]

            self.test_X = self.X[test_idx]
            self.test_y = self.y[test_idx]
    

    def run(self):
        self.fit_model, self.scaler = train(self.train_X, self.train_y, self.method)
        y_pred = test(self.fit_model, self.train_X, self.train_y, self.scaler)
        y_pred = test(self.fit_model, self.test_X, self.test_y, self.scaler)
        y_pred = test(self.fit_model, self.X, self.y, self.scaler)
        
        results = pd.DataFrame([])
        results['appName'] = self.gpudata['appName']
        results['coreF'] = self.gpudata['coreF']
        results['memF'] = self.gpudata['memF']
        results['avg_power'] = self.gpudata['power/W']
        results['modelled_power'] = None
        for idx, item in self.gpudata.iterrows():
            results.loc[idx, 'modelled_power'] = float(y_pred[idx] * self.gpudata[(self.gpudata['appName'] == item['appName']) & (self.gpudata['coreF'] == self.GPUCONF.CORE_FREQ) & (self.gpudata['memF'] == self.GPUCONF.MEM_FREQ)]['power/W'])
        return results


def main(opt):

    bench_conf = opt.benchmark_setting
    kernel_conf = opt.kernel_setting
    method = opt.method

    gpucard = bench_conf.split('-')[0]
    gpu_power_model = GPU_Power(gpucard, bench_conf, kernel_conf, method)

    gpu_power_model.split_data("kernel", 0.4)
    results = gpu_power_model.run()
    results.to_csv("csvs/ml/%s-%s-%s-Power.csv" % (bench_conf, kernel_conf, method))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=str, help='data file path', default='raw')
    parser.add_argument('--benchmark-setting', type=str, help='gpu and dvfs setting', default='p100-dvfs')
    parser.add_argument('--kernel-setting', type=str, help='kernel list', default='real')
    parser.add_argument('--method', type=str, help='analytical modeling method', default='svr-poly')
    
    opt = parser.parse_args()
    print opt

    main(opt)

