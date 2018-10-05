import pandas as pd
import numpy as np
import sys,os
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

highest_core = 1500
lowest_core = 700
highest_mem = 3900
lowest_mem = 2100

OUTPUT_PATH = 'figures'

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
    params['n_flop_sp'] = df['flop_count_sp'] * 1.0 / (df['warps'] * 32) 
    params['n_flop_dp'] = df['flop_count_dp'] * 1.0 / (df['warps'] * 32)
    params['n_int'] = df['inst_integer'] * 1.0 / (df['warps'] * 32) / 4.0 # for vision effect
    
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

def plot_inst_distribution(gpucard, csv_perf, save_filename = None):

    df = pd.read_csv(csv_perf, header = 0)

    in_kernels.sort()
    df = df[df.appName.isin(in_kernels)]
    df = df.reset_index(drop=True)
    df = df.sort_values(by = 'appName')
    
    print in_kernels
    print df

    norm_insts = inst_extract_and_normalize(df)

    fig, ax = plt.subplots(figsize = (8, 4))
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
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('')
    #margin = ax.get_ylim()[1]/4
    #ax.set_ylim(top=ax.get_ylim()[1]+margin)
    ax.set_xticks(x_axis + bar_width / 2)
    ax.set_xticklabels(map(get_abbr, in_kernels), size='large', rotation=0)

    ax.legend(fontsize='large', loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol = min(4, len(norm_insts.keys())))
    fig.subplots_adjust(top=0.7)
    ax.grid(linestyle=':')

    if not save_filename: # or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_line(selected_df, sorted_key='coreF', save_filename=None):

    fig, ax = plt.subplots(figsize = (8, 6))
    x_axis = list(selected_df[sorted_key].drop_duplicates())
    x_axis.sort()
    lines = []
    for idx, kernel in enumerate(in_kernels):
        tmp_data = selected_df[selected_df.appName == kernel]
        tmp_data = list(tmp_data.sort_values(by = [sorted_key])['time/ms'])
        tmp_data = [tmp_data[0] / item for item in tmp_data]
        print tmp_data
        kl_abbr = get_abbr(kernel)
        lines.append(ax.plot(x_axis, tmp_data, linewidth = 1.5, color = COLORS[idx], marker = MARKERS[idx], markersize = 10, label = kernel+"(%s)" % kl_abbr))

    ax.set_ylabel("Speed Up", size = 'x-large')
    ymax = ax.get_ylim()[1] * 1.1
    ymin = ax.get_ylim()[0] * 0.95
    ax.set_ylim(top = ymax, bottom = ymin)
    ax.yaxis.set_tick_params(labelsize=16)

    ax.set_xlim(min(x_axis) - 100, max(x_axis) + 100)
    ax.xaxis.set_tick_params(labelsize=16)

    ax.grid()
    ax.legend(fontsize='large', loc='upper left')

    if not save_filename: # or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')


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

    # fix core as lowest, scaling memory
    selected_df = df[df.coreF == lowest_core]
    plot_line(selected_df, 'memF', '%s_core_%d_mem_scaling' % (gpucard, lowest_core))

    selected_df = df[df.coreF == highest_core]
    plot_line(selected_df, 'memF', '%s_core_%d_mem_scaling' % (gpucard, highest_core))

    selected_df = df[df.memF == lowest_mem]
    plot_line(selected_df, 'coreF', '%s_mem_%d_core_scaling' % (gpucard, lowest_mem))

    selected_df = df[df.memF == highest_mem]
    plot_line(selected_df, 'coreF', '%s_mem_%d_core_scaling' % (gpucard, highest_mem))

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

def plot_perf_acc_analytical(gpu, save_filename = None):

    csv_file = "csvs/analytical/%s-kernel-aver.csv" % gpu
    df = pd.read_csv(csv_file, header = 0)
    print df.tail(3)

    kernels = list(df['kernel'])

    fig, ax = plt.subplots(figsize = (24, 4))
    # ax.title("Instruction Distribution")

    bar_width = 0.8
    x_axis = np.arange(len(df)) * bar_width * 4 + bar_width / 2

    fsize = 28
    ax.bar(x_axis, df['ape'] * 100, bar_width, label='mode 1', color=COLORS[2], hatch=HATCHES[1])

    ax.set_ylabel('Absolute Relative Error (%)', fontsize=fsize)
    ax.yaxis.set_tick_params(labelsize=fsize)
    ax.set_xlabel('')
    ax.set_ylim(top=100)
    ax.set_xticks(x_axis + bar_width * 1.5)
    ax.set_xticklabels(map(get_abbr, kernels), fontsize=fsize, rotation=90)

    ax.legend(fontsize=fsize, loc='upper center', ncol = 3, bbox_to_anchor=(0.5, 1.3))
    ax.grid(linestyle=':')

    if not save_filename: # or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_perf_acc_kernel(gpu, ml_algo, save_filename = None):

    csv_file = "csvs/ml/%s_%s_kernel_mode1.csv" % (gpu, ml_algo)
    m1_df = pd.read_csv(csv_file, header = 0)[:-1]

    csv_file = "csvs/ml/%s_%s_kernel_mode2.csv" % (gpu, ml_algo)
    m2_df = pd.read_csv(csv_file, header = 0)[:-1]

    csv_file = "csvs/ml/%s_%s_kernel_mode3.csv" % (gpu, ml_algo)
    m3_df = pd.read_csv(csv_file, header = 0)[:-1]

    print m1_df.tail(3)
    print m2_df.tail(3)
    print m3_df.tail(3)

    kernels = list(m1_df['kernel'])

    fig, ax = plt.subplots(figsize = (24, 4))
    # ax.title("Instruction Distribution")

    bar_width = 0.8
    x_axis = np.arange(len(m1_df)) * bar_width * 4 + bar_width / 2

    fsize = 28
    ax.bar(x_axis, m1_df['m1_error'] * 100, bar_width, label='mode 1', color=COLORS[2], hatch=HATCHES[1])
    ax.bar(x_axis + bar_width, m2_df['m2_error'] * 100, bar_width, label='mode 2', color=COLORS[3], hatch=HATCHES[2])
    ax.bar(x_axis + 2 * bar_width, m3_df['m3_error'] * 100, bar_width, label='mode 3', color=COLORS[4], hatch=HATCHES[3])
    #ax.bar(x_axis, m1_df['m1_error'] * 100, bar_width, label='mode 1', color='yellow', hatch=HATCHES[1])
    #ax.bar(x_axis + bar_width, m2_df['m2_error'] * 100, bar_width, label='mode 2', color='gray', hatch=HATCHES[2])
    #ax.bar(x_axis + 2 * bar_width, m3_df['m3_error'] * 100, bar_width, label='mode 3', color='red', hatch=HATCHES[3])

    ax.set_ylabel('Absolute Relative Error (%)', fontsize=fsize)
    ax.yaxis.set_tick_params(labelsize=fsize)
    ax.set_xlabel('')
    ax.set_ylim(top=100)
    ax.set_xticks(x_axis + bar_width * 1.5)
    ax.set_xticklabels(map(get_abbr, kernels), fontsize=fsize, rotation=90)

    ax.legend(fontsize=fsize, loc='upper center', ncol = 3, bbox_to_anchor=(0.5, 1.3))
    ax.grid(linestyle=':')

    if not save_filename: # or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_perf_acc_dvfs(gpu, ml_algo, save_filename = None):

    #csv_file = "csvs/ml/%s_%s_dvfs.csv" % (gpu, ml_algo)
    csv_file = "csvs/analytical/%s-%s-dvfs.csv" % (gpu, ml_algo)
    df = pd.read_csv(csv_file, header = 0)
    #df = pd.read_csv(csv_file, header = 0)[:-1]

    df['error'] = abs(df['real'] - df['predict']) / df['real'] * 100

    print df.tail(3)

    kernels = list(df['kernel'].drop_duplicates())
    dvfs_errs = []

    for kernel in kernels:
        tmp_errs = list(df[df.kernel == kernel]['error'])
        dvfs_errs.append(tmp_errs)

    fig, ax = plt.subplots(figsize = (16, 4))
    # ax.title("Instruction Distribution")

    bar_width = 0.8
    x_axis = np.arange(len(df)) * bar_width * 4 + bar_width / 2

    fsize = 18

    ax.boxplot(dvfs_errs)

    ax.set_ylabel('Absolute Relative Error (%)', fontsize=fsize)
    ax.yaxis.set_tick_params(labelsize=fsize)
    ax.set_xlabel('')
    ax.set_xticklabels(map(get_abbr, kernels), fontsize=fsize, rotation=90)
    ax.yaxis.grid(True)

    fig.subplots_adjust(bottom=0.3)

    if not save_filename: # or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

if __name__ == '__main__':

    ## gpu card and data file
    #gpu = 'gtx980'
    #version = 'real-small-workload'
    #ml_algo = 'svr-poly'

    #csv_file = "csvs/%s-dvfs-%s-Performance.csv" % (gpu, version)
    #plot_dvfs_scaling(gpu, csv_file)

    #csv_file = "csvs/%s-%s-Performance.csv" % (gpu, version)
    #plot_inst_distribution(gpu, csv_file, 'gtx980_sample_inst_dist')

    #ml_algo = 'xgboost'
    #gpu = 'gtx980'
    ## plot_perf_acc_kernel(gpu, ml_algo, '%s_%s_kernel' % (gpu, ml_algo))
    #plot_perf_acc_dvfs(gpu, ml_algo, '%s_%s_dvfs' % (gpu, ml_algo))
    #gpu = 'titanx'
    ## plot_perf_acc_kernel(gpu, ml_algo, '%s_%s_kernel' % (gpu, ml_algo))
    #plot_perf_acc_dvfs(gpu, ml_algo, '%s_%s_dvfs' % (gpu, ml_algo))
    #gpu = 'p100'
    ## plot_perf_acc_kernel(gpu, ml_algo, '%s_%s_kernel' % (gpu, ml_algo))
    #plot_perf_acc_dvfs(gpu, ml_algo, '%s_%s_dvfs' % (gpu, ml_algo))

    gpu = 'gtx980'
    plot_perf_acc_dvfs(gpu, 'qiang2018', '%s_analytical' % gpu)
    gpu = 'gtx1080ti'
    plot_perf_acc_dvfs(gpu, 'qiang2018', '%s_analytical' % gpu)
    gpu = 'p100'
    plot_perf_acc_dvfs(gpu, 'qiang2018', '%s_analytical' % gpu)

