import pandas as pd
import numpy as np
import sys
import random
from settings import *
# from sklearn import cross_validation
import matplotlib.pyplot as plt

in_kernels = ['BlackScholes', 'matrixMulShared']
out_kernels = ['binomialOptions', 'eigenvalues', 'scanUniformUpdate', 'stereoDisparity', 'reduction', 'matrixMulGlobal', 'cfd', 'hotspot', 'dxtc', 'backpropBackward']

# gpu card and data file
gpu = 'titanx-dvfs'
version = 'real'
csv_file = "csvs/%s-%s-Performance.csv" % (gpu, version)

def plot_dvfs_scaling(gpucard, csv_perf):

    if 'gtx980' in gpucard:
        GPUCONF = GTX980()
    elif 'p100' in gpucard:
        GPUCONF = P100()
    elif 'titanx' in gpucard:
        GPUCONF = TITANX()

    df = pd.read_csv(csv_perf, header = 0)

    df = df[df.appName.isin(in_kernels)]
    df = df.reset_index(drop=True)

    highest_core = 2009
    lowest_core = 1609
    highest_mem = 5013
    lowest_mem = 3513

    # fix core as lowest, scaling memory
    selected_df = df[df.coreF == lowest_core]

    plt.title("Performance Scaling with varying memory frequency(core frequency = 1609)")
    x_axis = list(selected_df['memF'].drop_duplicates())
    x_axis.sort()
    for kernel in in_kernels:
        tmp_data = selected_df[selected_df.appName == kernel]
        tmp_data = list(tmp_data.sort_values(by = ['memF'])['time/ms'])
        tmp_data = [tmp_data[0] / item for item in tmp_data]
        print tmp_data
        plt.plot(x_axis, tmp_data)

    plt.show()

    # fix core as highest, scaling memory 
    selected_df = df[df.coreF == highest_core]

    plt.title("Performance Scaling with varying memory frequency(core frequency = 1609)")
    x_axis = list(selected_df['memF'].drop_duplicates())
    x_axis.sort()
    for kernel in in_kernels:
        tmp_data = selected_df[selected_df.appName == kernel]
        tmp_data = list(tmp_data.sort_values(by = ['memF'])['time/ms'])
        tmp_data = [tmp_data[0] / item for item in tmp_data]
        print tmp_data
        plt.plot(x_axis, tmp_data)

    plt.show()

    # fix mem as lowest, scaling core
    selected_df = df[df.memF == lowest_mem]
    selected_df = selected_df.sort_values(by = ['appName', 'coreF'])

    plt.title("Performance Scaling with varying memory frequency(core frequency = 1609)")
    x_axis = list(selected_df['coreF'].drop_duplicates())
    x_axis.sort()
    for kernel in in_kernels:
        tmp_data = selected_df[selected_df.appName == kernel]
        tmp_data = list(tmp_data.sort_values(by = ['coreF'])['time/ms'])
        tmp_data = [tmp_data[0] / item for item in tmp_data]
        print tmp_data
        plt.plot(x_axis, tmp_data)

    plt.show()

    # fix mem as highest, scaling core 
    selected_df = df[df.memF == highest_mem]
    selected_df = selected_df.sort_values(by = ['appName', 'coreF'])

    plt.title("Performance Scaling with varying memory frequency(core frequency = 1609)")
    x_axis = list(selected_df['coreF'].drop_duplicates())
    x_axis.sort()
    for kernel in in_kernels:
        tmp_data = selected_df[selected_df.appName == kernel]
        tmp_data = list(tmp_data.sort_values(by = ['coreF'])['time/ms'])
        tmp_data = [tmp_data[0] / item for item in tmp_data]
        print tmp_data
        plt.plot(x_axis, tmp_data)

    plt.show()

    ##params = pd.DataFrame(columns=['n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'n_dm_ld', 'n_dm_st', 'n_flop_sp', 'mem_insts', 'insts']) 
    #params = pd.DataFrame(columns=['n_gld', 'n_gst', 'gld_trans_per_req', 'gst_trans_per_req', \
    #    			   'n_dm_ld', 'n_dm_st', \
    #    			   'n_l2_ld', 'n_l2_st', \
    #    			   'n_shm_ld', 'n_shm_st', 'shld_trans_per_req', 'shst_trans_per_req', \
    #    			   'tex_hit_rate', 'tex_trans', \
    #    			   'n_flop_sp', 'n_flop_sp_fma', 'n_flop_sp_spec', 'n_flop_dp', 'n_flop_dp_fma', 'n_int', \
    #    			   ]) 
    #
    ## hardware parameters
    #df['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    #
    ## global memory information
    #params['n_gld'] = df['gld_transactions'] / df['warps'] 
    #params['n_gst'] = df['gst_transactions'] / df['warps']
    #params['gld_trans_per_req'] = df['gld_transactions_per_request'] 
    #params['gst_trans_per_req'] = df['gst_transactions_per_request']
    #params['gld_req'] = 0
    #params.loc[params['gld_trans_per_req'] > 0, 'gld_req'] = params.loc[params['gld_trans_per_req'] > 0, 'n_gld'] / params.loc[params['gld_trans_per_req'] > 0, 'gld_trans_per_req']
    #params['gst_req'] = 0
    #params.loc[params['gst_trans_per_req'] > 0, 'gst_req'] = params.loc[params['gst_trans_per_req'] > 0, 'n_gst'] / params.loc[params['gst_trans_per_req'] > 0, 'gst_trans_per_req']
    #params['n_gm'] = params['n_gld'] + params['n_gst']
    #params['gm_req'] = params['gld_req'] + params['gst_req']
    #
    ## dram memory information
    #params['n_dm_ld'] = df['dram_read_transactions'] / df['warps']
    #params['n_dm_st'] = df['dram_write_transactions'] / df['warps']
    #params['n_dm'] = params['n_dm_ld'] + params['n_dm_st'] 
    #
    ## l2 cache information
    #params['n_l2_ld'] = df['l2_read_transactions'] / df['warps']
    #params['n_l2_st'] = df['l2_write_transactions'] / df['warps']
    #params['n_l2'] = params['n_l2_ld'] + params['n_l2_st']
    #
    ## shared memory information
    #params['n_shm_ld'] = df['shared_load_transactions'] / df['warps'] 
    #params['n_shm_st'] = df['shared_store_transactions'] / df['warps'] 
    #params['shld_trans_per_req'] = df['shared_load_transactions_per_request']
    #params['shst_trans_per_req'] = df['shared_store_transactions_per_request'] 
    #params['shld_req'] = 0
    #params.loc[params['shld_trans_per_req'] > 0, 'shld_req'] = params.loc[params['shld_trans_per_req'] > 0, 'n_shm_ld'] / params.loc[params['shld_trans_per_req'] > 0, 'shld_trans_per_req']
    #params['shst_req'] = 0
    #params.loc[params['shst_trans_per_req'] > 0, 'shst_req'] = params.loc[params['shst_trans_per_req'] > 0, 'n_shm_st'] / params.loc[params['shst_trans_per_req'] > 0, 'shst_trans_per_req']
    #params['n_shm'] = params['n_shm_ld'] + params['n_shm_st'] 
    #params['shm_req'] = params['shst_req'] + params['shld_req']
    #
    ## texture memory information
    #params['tex_hit_rate'] = df['tex_cache_hit_rate']
    #params['tex_trans'] = df['tex_cache_transactions'] / df['warps']

    ## compute insts
    #params['n_flop_sp'] = df['flop_count_sp'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    #params['n_flop_sp_fma'] = df['flop_count_sp_fma'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    #params['n_flop_sp_spec'] = df['flop_count_sp_special'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    #params['n_flop_dp'] = df['flop_count_dp'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    #params['n_flop_dp_fma'] = df['flop_count_dp_fma'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    #params['n_int'] = df['inst_integer'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM

    ## branch
    #params['branch'] = df['cf_executed'] * 1.0 / (df['warps'])

    ## instruction statistic
    #params['inst_per_warp'] = df['inst_per_warp']

    ### other parameters
    ##df['mem_insts'] = params['n_gld'] + params['n_gst'] + params['n_shm_ld'] + params['n_shm_st']
    ##params['other_insts'] = (df['inst_per_warp'] - df['mem_insts'] - params['n_flop_sp']) * 1.0 # / GPUCONF.CORES_SM
    ##params.loc[params['other_insts'] < 0, 'other_insts'] = 0
    ### print params['other_insts']
    #
    ## grouth truth cycle per SM per round / ground truth IPC
    #params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy']))
    ##params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT))
    ##params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / df['warps']
    ##try:
    ##    params['real_cycle'] = df['ipc']
    ##except Exception as e:
    ##    params['real_cycle'] = df['executed_ipc']

    ##params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy'])) / params['inst_per_warp']
    ##print params['real_cycle']
    #
    ## hardware information, frequency ratio, core/mem
    #params['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    #params['act_util'] = df['achieved_occupancy']
    #
    ## select features for training
    ##inst_features = ['n_dm', 'n_l2', 'n_shm', 'tex_trans', 'n_flop_sp', 'n_flop_dp', 'n_int', 'branch']
    #inst_features = ['n_dm', 'n_l2', 'n_shm', 'tex_trans', 'n_flop_sp', 'n_flop_dp', 'n_int']

    ## normalized with inst_per_warp, predict cycle per round
    #X = params.loc[:, inst_features]
    #X = X.div(params['inst_per_warp'], axis=0)
    #y = params['real_cycle'] / params['inst_per_warp']

    ### normalized with total amount of insts, predict cycle per round
    ##X = params.loc[:, inst_features]
    ##y = params['real_cycle'].div(X.loc[:, :].sum(axis=1), axis=0) # for real cycle
    ##X = X.div(X.loc[:, :].sum(axis=1), axis=0)

    ### normalized with total amount of insts, predict ipc
    ##X = params.loc[:, inst_features]
    ##X = X.div(X.loc[:, :].sum(axis=1), axis=0)
    ##y = params['real_cycle']

    #util_features = ['act_util', 'c_to_m']
    #for uf in util_features:
    #    X[uf] = params[uf]
    #
    #print "Total number of samples:", len(X)
    #X = X.astype(np.float64)
    ##print X.head(5)
    ##print y.head(5)

    #features = pd.DataFrame([])
    #features['appName'] = df['appName']
    #for f in X.keys():
    #    features[f] = X[f]
    #features['inst_per_warp'] = params['inst_per_warp']
    #features['real_cycle'] = y
    #features.to_csv("csvs/%s-%s-features.csv" % (gpucard, version))

    #return X, y, df

if __name__ == '__main__':

    plot_dvfs_scaling(gpu, csv_file)
