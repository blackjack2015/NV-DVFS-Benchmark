import pandas as pd
import numpy as np
import sys
import random
from settings import *
# from sklearn import cross_validation
import matplotlib.pyplot as plt

MARKERS = ['^', '<', 'o', 's']
HATCHES = ['//', '--', '\\\\', '||', '++', '--', '..', '++', '\\\\']
COLORS = ['#2F4F4F', '#808080', '#A9A9A9', '#778899', '#DCDCDC', '#556677', '#1D3E3E', '#808080', '#DCDCDC']
# COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

in_kernels = ['BlackScholes', 'matrixMulShared', 'backpropForward', 'convolutionSeparable']
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

    fig, ax = plt.subplots(figsize = (8, 6))
    # ax.title("Instruction Distribution")

    bar_width = 0.5
    x_axis = np.arange(len(in_kernels)) * bar_width * 2 + bar_width / 2
    lastInst = np.zeros(len(in_kernels))
    for idx, key in enumerate(norm_insts.keys()):
        tmp_data = norm_insts[key]
        print tmp_data
        ax.bar(x_axis, tmp_data, bar_width, bottom=lastInst, label=key, color=COLORS[idx % len(COLORS)], hatch=HATCHES[idx % len(HATCHES)])
        lastInst += tmp_data

    ax.set_ylabel('Percentage', size='x-large')
    ax.set_xlabel('')
    #margin = ax.get_ylim()[1]/4
    #ax.set_ylim(top=ax.get_ylim()[1]+margin)
    ax.set_xticks(x_axis + bar_width / 2)
    ax.set_xticklabels(in_kernels, size='medium', rotation=0)

    ax.legend(fontsize='medium', loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol = min(4, len(norm_insts.keys())))
    fig.subplots_adjust(top=0.88)
    ax.grid(linestyle=':')
    plt.show()

def plot_line(selected_df, sorted_key='coreF'):

    fig, ax = plt.subplots(figsize = (8, 6))
    x_axis = list(selected_df[sorted_key].drop_duplicates())
    x_axis.sort()
    lines = []
    for idx, kernel in enumerate(in_kernels):
        tmp_data = selected_df[selected_df.appName == kernel]
        tmp_data = list(tmp_data.sort_values(by = [sorted_key])['time/ms'])
        tmp_data = [tmp_data[0] / item for item in tmp_data]
        print tmp_data
        lines.append(ax.plot(x_axis, tmp_data, linewidth = 1.5, color = COLORS[idx], marker = MARKERS[idx], markersize = 10, label = kernel))

    ax.grid()
    ax.legend(fontsize='medium', loc='upper left')
    plt.show()

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
    plot_line(selected_df, 'memF')

    selected_df = df[df.coreF == highest_core]
    plot_line(selected_df, 'memF')

    selected_df = df[df.memF == highest_mem]
    plot_line(selected_df, 'coreF')

    selected_df = df[df.memF == lowest_mem]
    plot_line(selected_df, 'coreF')

    ## fix core as highest, scaling memory 
    #selected_df = df[df.coreF == highest_core]

    #plt.title("Performance Scaling with varying memory frequency(core frequency = 1609)")
    #x_axis = list(selected_df['memF'].drop_duplicates())
    #x_axis.sort()
    #for idx, kernel in enumerate(in_kernels):
    #    tmp_data = selected_df[selected_df.appName == kernel]
    #    tmp_data = list(tmp_data.sort_values(by = ['memF'])['time/ms'])
    #    tmp_data = [tmp_data[0] / item for item in tmp_data]
    #    print tmp_data
    #    plt.plot(x_axis, tmp_data, color = COLORS[idx], marker = MARKERS[idx], label = kernel)

    #plt.show()

    ## fix mem as lowest, scaling core
    #selected_df = df[df.memF == lowest_mem]
    #selected_df = selected_df.sort_values(by = ['appName', 'coreF'])

    #plt.title("Performance Scaling with varying memory frequency(core frequency = 1609)")
    #x_axis = list(selected_df['coreF'].drop_duplicates())
    #x_axis.sort()
    #for idx, kernel in enumerate(in_kernels):
    #    tmp_data = selected_df[selected_df.appName == kernel]
    #    tmp_data = list(tmp_data.sort_values(by = ['coreF'])['time/ms'])
    #    tmp_data = [tmp_data[0] / item for item in tmp_data]
    #    print tmp_data
    #    plt.plot(x_axis, tmp_data, color = COLORS[idx], marker = MARKERS[idx])

    #plt.show()

    ## fix mem as highest, scaling core 
    #selected_df = df[df.memF == highest_mem]
    #selected_df = selected_df.sort_values(by = ['appName', 'coreF'])

    #plt.title("Performance Scaling with varying memory frequency(core frequency = 1609)")
    #x_axis = list(selected_df['coreF'].drop_duplicates())
    #x_axis.sort()
    #for idx, kernel in enumerate(in_kernels):
    #    tmp_data = selected_df[selected_df.appName == kernel]
    #    tmp_data = list(tmp_data.sort_values(by = ['coreF'])['time/ms'])
    #    tmp_data = [tmp_data[0] / item for item in tmp_data]
    #    print tmp_data
    #    plt.plot(x_axis, tmp_data, color = COLORS[idx], marker = MARKERS[idx])

    #plt.show()

if __name__ == '__main__':

    plot_dvfs_scaling(gpu, csv_file)
    #plot_inst_distribution(gpu, csv_file)
