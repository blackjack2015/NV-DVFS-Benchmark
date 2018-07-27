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

def inst_extract_and_normalize(df):

    params = pd.DataFrame(columns=[])

    # global memory information
    params['n_gld'] = df['gld_transactions'] / df['warps'] 
    params['n_gst'] = df['gst_transactions'] / df['warps']
    params['n_gm'] = params['n_gld'] + params['n_gst']
    
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
    params['n_shm'] = params['n_shm_ld'] + params['n_shm_st'] 
    
    # texture memory information
    params['tex_trans'] = df['tex_cache_transactions'] / df['warps']
    
    # compute insts
    params['n_flop_sp'] = df['flop_count_sp'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    params['n_flop_dp'] = df['flop_count_dp'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    params['n_int'] = df['inst_integer'] * 1.0 / (df['warps'] * 32) # / GPUCONF.CORES_SM
    
    # instruction statistic
    params['inst_per_warp'] = df['inst_per_warp']
    
    # select features for training
    inst_features = ['n_dm', 'n_l2', 'n_shm', 'tex_trans', 'n_flop_sp', 'n_flop_dp', 'n_int']
    
    ## normalized with inst_per_warp, predict cycle per round
    #X = params.loc[:, inst_features]
    #X = X.div(params['inst_per_warp'], axis=0)
    
    # normalized with total amount of insts, predict cycle per round
    X = params.loc[:, inst_features]
    X = X.div(X.loc[:, :].sum(axis=1), axis=0)
    
    X = X.astype(np.float64)

    return X

def plot_inst_distribution(gpucard, csv_perf):

    df = pd.read_csv(csv_perf, header = 0)

    df = df[df.appName.isin(in_kernels)]
    df = df.reset_index(drop=True)

    norm_insts = inst_extract_and_normalize(df)


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

if __name__ == '__main__':

    plot_dvfs_scaling(gpu, csv_file)
