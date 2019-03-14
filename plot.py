import pandas as pd
import numpy as np
import sys,os
import random
from settings import *
# from sklearn import cross_validation
import matplotlib.pyplot as plt
from scipy.stats.stats import pearsonr 
#import seaborn as sns
#from matplotlib_colorbar.colorbar import Colorbar

MARKERS = ['^', '<', 'o', 's']
HATCHES = ['//', '--', '\\\\', '||', '++', '--', '..', '++', '\\\\']
GRAYS = ['#2F4F4F', '#808080', '#A9A9A9', '#778899', '#DCDCDC', '#556677', '#1D3E3E', '#808080', '#DCDCDC']
COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']

in_kernels = ['BlackScholes', 'matrixMulShared', 'backpropForward', 'histogram']
#in_kernels = ['BlackScholes', 'matrixMul', 'backprop', 'convolutionSeparable']
out_kernels = ['binomialOptions', 'eigenvalues', 'scanUniformUpdate', 'stereoDisparity', 'reduction', 'matrixMulGlobal', 'cfd', 'hotspot', 'dxtc', 'backpropBackward']
# experimental test
pointer = ['convolutionTexture', 'nn', 'SobolQRNG', 'reduction', 'hotspot'] 
#pointer = []
extras = ['backpropBackward', 'binomialOptions', 'cfd', 'eigenvalues', 'gaussian', 'srad', 'dxtc', 'pathfinder', 'scanUniformUpdate', 'stereoDisparity'] 
#extras = []


highest_core = 1000
lowest_core = 500
highest_mem = 1000
lowest_mem = 500

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

    #in_kernels.sort()
    #df = df[df.appName.isin(in_kernels) & (df.coreF==highest_core) & (df.memF==highest_mem)]
    df = df[~df.appName.isin(extras) & (df.coreF==highest_core) & (df.memF==highest_mem)]
    #df = df[(df.coreF==highest_core) & (df.memF==highest_mem)]
    df = df.reset_index(drop=True)
    df = df.sort_values(by = 'appName')
    in_kernels = list(df.appName.drop_duplicates())
    
    print in_kernels
    #print df

    norm_insts = inst_extract_and_normalize(df)

    fig, ax = plt.subplots(figsize = (16, 4))
    # ax.title("Instruction Distribution")

    bar_width = 0.5
    x_axis = np.arange(len(in_kernels)) * bar_width * 2 + bar_width / 2
    lastInst = np.zeros(len(in_kernels))
    for idx, key in enumerate(norm_insts.keys()):
        tmp_data = norm_insts[key]
        #print tmp_data
        ax.bar(x_axis, tmp_data, bar_width, alpha=0.8, bottom=lastInst, label=key, color=GRAYS[idx % len(COLORS)], hatch=HATCHES[idx % len(HATCHES)])
        lastInst += tmp_data

    ax.set_ylabel('Percentage', size='x-large')
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_xlabel('')
    #margin = ax.get_ylim()[1]/4
    #ax.set_ylim(top=ax.get_ylim()[1]+margin)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(map(get_abbr, in_kernels), size='large', rotation=0)

    ax.legend(fontsize='large', loc='upper center', bbox_to_anchor=(0.5, 1.35), ncol = min(4, len(norm_insts.keys())))
    fig.subplots_adjust(top=0.7)
    ax.grid(linestyle=':')

    if not save_filename:# or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_line(selected_df, sorted_key='coreF', fixed_freq=500, save_filename=None):

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
        #lines.append(ax.plot(x_axis, tmp_data, linewidth = 1.5, color = COLORS[idx], marker = MARKERS[idx], markersize = 14, markerfacecolor = 'none', label = kernel+"(%s)" % kl_abbr))
        lines.append(ax.plot(x_axis, tmp_data, linewidth = 1.5, color = COLORS[idx], marker = MARKERS[idx], markersize = 14, markeredgecolor='k', markerfacecolor = 'none', label = kernel+"(%s)" % kl_abbr))

    ax.set_ylabel("Speed Up", size = 24)
    if sorted_key == 'coreF':
        ax.set_xlabel("$f^{MEM}=$%d MHz, Core Frequency/MHz" % fixed_freq, size = 24)
    else:
        ax.set_xlabel("$f^{SM}=$%d MHz, Memory Frequency/MHz" % fixed_freq, size = 24)
    ymax = ax.get_ylim()[1] * 1.15
    ymin = ax.get_ylim()[0] * 0.95
    ax.set_ylim(top = ymax, bottom = ymin)
    ax.yaxis.set_tick_params(labelsize=24)

    ax.set_xlim(min(x_axis) - 100, max(x_axis) + 100)
    ax.xaxis.set_tick_params(labelsize=24)

    ax.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    ax.legend(fontsize=18, loc='upper left')

    if not save_filename:# or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_dvfs_roofline(gpu, version, kernel, show=False, save_filename=None):

    #csv_file = "csvs/ml/%s_%s_dvfs.csv" % (gpu, ml_algo)
    csv_file = "csvs/analytical/cycles/%s-%s-qiang2018-cycles.csv" % (gpu, version)
    df = pd.read_csv(csv_file, header = 0)

    fig, ax = plt.subplots(figsize = (8, 6))

    df = df[df.appName == kernel]

    x_axis = list(df['c_to_m'])
    x_axis.sort()
    lines = []

    x_axis = x_axis[::4]
    #x_axis = x_axis
    tmp_data = list(df.sort_values(by = ['c_to_m'])['real_cycle'])[::4]
    lines.append(ax.plot(x_axis, tmp_data, linewidth = 1.5, color = COLORS[1], marker = MARKERS[1], markersize = 16, markeredgecolor = 'k', label = 'measured cycles', markerfacecolor='none'))
    tmp_data = list(df.sort_values(by = ['c_to_m'])['mem_del'])[::4]
    lines.append(ax.plot(x_axis, tmp_data, linewidth = 1.5, color = COLORS[2], marker = MARKERS[2], markersize = 16, markeredgecolor = 'k', label = 'FULL_MEM cycles', markerfacecolor='none'))
    tmp_data = list(df.sort_values(by = ['c_to_m'])['sm_del'])[::4]
    lines.append(ax.plot(x_axis, tmp_data, linewidth = 1.5, color = COLORS[3], marker = MARKERS[3], markersize = 16, markeredgecolor = 'k', label = 'FULL_COMP cycles', markerfacecolor='none'))

    ax.set_ylabel("Cycles", size = 24)
    ymax = ax.get_ylim()[1] * 1.35
    ymin = ax.get_ylim()[0] * 0.65
    ax.set_ylim(top = ymax, bottom = ymin)
    ax.yaxis.set_tick_params(labelsize=24)
    ax.ticklabel_format(style='sci', scilimits=(1,1), axis='y')
    ax.yaxis.offsetText.set_fontsize(20)

    ax.set_xlabel("$f^{SM}/f^{MEM}$", size = 24)
    #ax.set_xlim(min(x_axis) - 100, max(x_axis) + 100)
    ax.xaxis.set_tick_params(labelsize=24)

    ax.grid(color='#5e5c5c', linestyle='-.', linewidth=1)
    ax.legend(fontsize=18, loc='upper left')

    if show:
        plt.show()
    if save_filename:# or True:
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
    selected_df = df[(df.coreF == lowest_core) & (df.memF >= lowest_mem)] 
    plot_line(selected_df, 'memF', lowest_core, '%s_core_%d_mem_scaling' % (gpucard, lowest_core))

    # fix core as highest, scaling memory
    selected_df = df[(df.coreF == highest_core) & (df.memF >= lowest_mem)]
    plot_line(selected_df, 'memF', highest_core, '%s_core_%d_mem_scaling' % (gpucard, highest_core))

    # fix memory as lowest, scaling core
    selected_df = df[(df.memF == lowest_mem) & (df.coreF >= lowest_core)]
    plot_line(selected_df, 'coreF', lowest_mem, '%s_mem_%d_core_scaling' % (gpucard, lowest_mem))

    # fix memory as highest, scaling core
    selected_df = df[(df.memF == highest_mem) & (df.coreF >= lowest_core)]
    plot_line(selected_df, 'coreF', highest_mem, '%s_mem_%d_core_scaling' % (gpucard, highest_mem))

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

def plot_perf_acc_freq_merge(save_filename = None):

    fig, axes = plt.subplots(nrows = 3, ncols = 1, figsize = (9, 9), gridspec_kw = {'height_ratios':[4, 4, 1]})
    ax_size = 18
    cmap_str = 'gist_rainbow'

    # read gtx980
    csv_file = "csvs/analytical/results/gtx980-low-dvfs-real-small-workload-qiang2018-dvfs.csv"
    df = pd.read_csv(csv_file, header = 0)
    print df.tail(3)

    df.error = df.error * 100.0
    #freq_error = df['error'].groupby([df['coreF'], df['memF']]).mean()
    piv = pd.pivot_table(df, values="error",index=["coreF"], columns=["memF"], fill_value=0)

    print piv

    ax = axes[0]
    im = ax.imshow(piv, cmap = cmap_str, origin="lower", aspect = 'auto', vmin=4, vmax=12)

    data = piv.values
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x , y, '%.2f' % (data[y, x]), #data[y,x] +0.05 , data[y,x] + 0.05
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=ax_size,
                 color='k')

    ax.set_xticks(range(len(piv.columns)))
    ax.set_yticks(range(len(piv.index)))
    ax.set_xticklabels(piv.columns, size=ax_size)
    ax.set_yticklabels(piv.index, size=ax_size)
    ax.set_xlabel("Memory Frequency/MHz", size=ax_size)
    ax.set_ylabel("Core Frequency/MHz", size=ax_size)
    
    # read gtx1080ti
    csv_file = "csvs/analytical/results/gtx1080ti-dvfs-real-qiang2018-dvfs.csv"
    df = pd.read_csv(csv_file, header = 0)
    print df.tail(3)

    df.error = df.error * 100.0
    #freq_error = df['error'].groupby([df['coreF'], df['memF']]).mean()
    piv = pd.pivot_table(df, values="error",index=["coreF"], columns=["memF"], fill_value=0)

    print piv

    ax = axes[1]
    im = ax.imshow(piv, cmap = cmap_str, origin="lower", aspect = 'auto', vmin=4, vmax=12)

    data = piv.values
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x , y, '%.2f' % (data[y, x]), #data[y,x] +0.05 , data[y,x] + 0.05
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=ax_size,
                 color='k')

    ax.set_xticks(range(len(piv.columns)))
    ax.set_yticks(range(len(piv.index)))
    ax.set_xticklabels(piv.columns, size=ax_size)
    ax.set_yticklabels(piv.index, size=ax_size)
    ax.set_xlabel("Memory Frequency/MHz", size=ax_size)
    ax.set_ylabel("Core Frequency/MHz", size=ax_size)

    # read p100
    csv_file = "csvs/analytical/results/p100-dvfs-real-qiang2018-dvfs.csv"
    df = pd.read_csv(csv_file, header = 0)
    print df.tail(3)

    df.error = df.error * 100.0
    #freq_error = df['error'].groupby([df['coreF'], df['memF']]).mean()
    piv = pd.pivot_table(df, values="error",index=["memF"], columns=["coreF"], fill_value=0)

    print piv

    ax = axes[2]
    im = ax.imshow(piv, cmap = cmap_str, origin="lower", aspect = 'auto', vmin=4, vmax=12)

    data = piv.values
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x , y, '%.2f' % (data[y, x]), #data[y,x] +0.05 , data[y,x] + 0.05
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=ax_size,
                 color='k')

    ax.get_yaxis().set_visible(False)
    ax.set_xticks(range(len(piv.columns)))
    #ax.set_yticks(range(len(piv.index)))
    ax.set_xticklabels(piv.columns, size=ax_size)
    #ax.set_yticklabels(piv.index, size=ax_size)
    ax.set_xlabel("Memory Frequency/MHz", size=ax_size)
    #ax.set_ylabel("Core Frequency/MHz", size=ax_size)

    fig.subplots_adjust(right=0.85, hspace = 0.62)
    # add an axes, lower left corner in [0.83, 0.1] measured in figure coordinate with axes width 0.02 and height 0.8
    cb_ax = fig.add_axes([0.9, 0.1, 0.04, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)

    #fig.subplots_adjust(right=1.2)
    #cbar_ax = fig.add_axes([0.9, 0.15, 0.05, 0.7])

    #ax.set_xticks(range(len(piv.columns)))
    #ax.set_yticks(range(len(piv.index)))
    #ax.set_xticklabels(piv.columns, size=24)
    #ax.set_yticklabels(piv.index, size=24)
    #ax.set_xlabel("Memory Frequency/MHz", size=26)
    #ax.set_ylabel("Core Frequency/MHz", size=26)
    
    if not save_filename:# or True:
        #plt.tight_layout()
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_perf_acc_freq(gpu, version, method, save_filename = None):

    csv_file = "csvs/analytical/results/%s-%s-%s-dvfs.csv" % (gpu, version, method)
    df = pd.read_csv(csv_file, header = 0)
    print df.tail(3)

    if 'p100' in gpu:
        fig, ax = plt.subplots(figsize = (16, 3))
    else:
        fig, ax = plt.subplots(figsize = (16, 6))

    df.error = df.error * 100.0
    #freq_error = df['error'].groupby([df['coreF'], df['memF']]).mean()
    if 'p100' in gpu:
        piv = pd.pivot_table(df, values="error",index=["memF"], columns=["coreF"], fill_value=0)
    else:
        piv = pd.pivot_table(df, values="error",index=["coreF"], columns=["memF"], fill_value=0)

    print piv

    error_min = piv.values.min()
    error_max = piv.values.max()

    im = ax.imshow(piv, cmap="Reds", origin="lower", aspect='auto', vmin=error_min * 0.82, vmax=error_max * 1.18)
    #im = ax.imshow(piv, cmap="gray", origin="lower", aspect='auto', vmin=error_min * 0.6, vmax=error_max * 1.2)
    if 'p100' in gpu:
        cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.4)
        #cbar = Colorbar(im, location='upper', orientation='horizontal')
    else:
        cbar = fig.colorbar(im, ax=ax)

    cbar.ax.tick_params(labelsize=24) 

    data = piv.values
    for y in range(data.shape[0]):
        for x in range(data.shape[1]):
            ax.text(x , y, '%.2f' % (data[y, x]), #data[y,x] +0.05 , data[y,x] + 0.05
                 horizontalalignment='center',
                 verticalalignment='center',
                 size=24,
                 color='k')

    ax.set_xticks(range(len(piv.columns)))
    ax.set_yticks(range(len(piv.index)))
    ax.set_xticklabels(piv.columns, size=24)
    ax.set_yticklabels(piv.index, size=24)
    ax.set_xlabel("Memory Frequency/MHz", size=26)
    ax.set_ylabel("Core Frequency/MHz", size=26)
    
    if not save_filename:# or True:
        plt.tight_layout()
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')



def plot_perf_acc_corr(gpu, version, method, save_filename = None):

    csv_file = "csvs/analytical/results/%s-%s-%s-dvfs.csv" % (gpu, version, method)
    df = pd.read_csv(csv_file, header = 0)
    print df.tail(3)

    # ax.title("Instruction Distribution")
    fig, ax = plt.subplots(figsize = (8, 6))
    lines = []

    x_axis = list(df.predict)
    y_axis = list(df.real)

    corrs = pearsonr(x_axis, y_axis)[0]
    aver_errs = np.mean([abs(x_axis[i] - y_axis[i])/y_axis[i] for i in range(len(x_axis))]) * 100

    lines.append(ax.scatter(x_axis, y_axis, linewidth = 1.5, color = 'b', alpha=0.5, marker = 'o', label = 'Modeling Cycles\n[Correl=%.3f, Err=%.3f]' % (corrs, aver_errs)))
    lines.append(ax.plot(x_axis, x_axis, linewidth = 1.5, color = 'r', label = 'Ground Truth'))
    fsize = 18

    ax.set_ylabel('Hardware Measured Cycles', fontsize=fsize)
    ax.set_yscale('log')
    ax.yaxis.set_tick_params(labelsize=fsize)
    #ymax = ax.get_ylim()[1] * 1.1
    #ax.set_ylim(top = ymax)
  
    ax.set_xscale('log')
    ax.set_xlabel('Modeling Cycles', fontsize=fsize)
    ax.xaxis.set_tick_params(labelsize=fsize)
    #ax.set_xticks(x_axis)

    ax.legend(fontsize=fsize, loc='upper left')
    ax.grid(linestyle=':')

    if not save_filename:# or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_perf_acc_analytical(gpu, version, method, save_filename = None):

    csv_file = "csvs/analytical/results/%s-%s-%s-aver.csv" % (gpu, version, method)
    df = pd.read_csv(csv_file, header = 0)
    print df.tail(3)

    kernels = list(df['kernel'])

    fig, ax = plt.subplots(figsize = (24, 4))
    # ax.title("Instruction Distribution")

    bar_width = 0.8
    x_axis = np.arange(len(df)) 

    fsize = 20
    ax.bar(x_axis, df['ape'] * 100, bar_width, color=COLORS[1])

    ax.set_ylabel('Absolute Relative Error (%)', fontsize=fsize)
    ax.yaxis.set_tick_params(labelsize=fsize)
    ax.set_xlabel('')
    ymax = ax.get_ylim()[1] * 1.1
    ax.set_ylim(top = ymax)
    ax.set_xticks(x_axis)
    ax.set_xticklabels(map(get_abbr, kernels), fontsize=fsize) #rotation=90)

    #ax.legend(fontsize=fsize, loc='upper center', ncol = 3, bbox_to_anchor=(0.5, 1.3))
    ax.grid(linestyle=':')

    if not save_filename:
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

    if not save_filename or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_perf_acc_dvfs(gpu, ml_algo, save_filename = None):

    #csv_file = "csvs/ml/%s_%s_dvfs.csv" % (gpu, ml_algo)
    csv_file = "csvs/analytical/results/%s-%s-dvfs.csv" % (gpu, ml_algo)
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
    ax.set_ylim(top=10)
    ax.set_xlabel('')
    ax.set_xticklabels(map(get_abbr, kernels), fontsize=fsize, rotation=90)
    ax.yaxis.grid(True)

    fig.subplots_adjust(bottom=0.3)

    if not save_filename or True:
        plt.show()
	return
    else:
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.pdf'%save_filename), bbox_inches='tight')
        plt.savefig(os.path.join(OUTPUT_PATH, '%s.png'%save_filename), bbox_inches='tight')

def plot_energy(gpu, version, save_filename = None):
    
    csv_file = "csvs/analytical/results/%s-%s-qiang2018-dvfs.csv" % (gpu, version)
    perf_data = pd.read_csv(csv_file, header = 0)
    csv_file = "csvs/ml/%s-%s-xgboost-Power.csv" % (gpu, version)
    pow_data = pd.read_csv(csv_file, header = 0)

    energy_data = pd.DataFrame([])

    kernelset = perf_data['kernel'].drop_duplicates().reset_index(drop=True)
    print kernelset

    energy_data['appName'] = kernelset
    energy_data['defaultE'] = None
    energy_data['bestE'] = None
    energy_data['bestC'] = None
    energy_data['bestM'] = None
    energy_data['predictE'] = None
    energy_data['predictC'] = None
    energy_data['predictM'] = None

    for idx, item in energy_data.iterrows():
        cur_app = item.appName
        cur_perf = perf_data[perf_data['kernel'] == cur_app]
        cur_pow = pow_data[pow_data['appName'] == cur_app]
        cur_perf = cur_perf.sort_values(by = ['kernel', 'coreF', 'memF']).reset_index(drop=True)
        cur_pow = cur_pow.sort_values(by = ['appName', 'coreF', 'memF']).reset_index(drop=True)

        cur_perf.real = cur_perf.real / 1.0e6 / cur_perf.coreF
        cur_perf.predict = cur_perf.predict / 1.0e6 / cur_perf.coreF
        measureE = cur_perf.real * cur_pow.avg_power
        modelledE = cur_perf.predict * cur_pow.modelled_power

        bestE = min(measureE)
        bestE_idx = np.argmin(measureE)
        bestC = cur_perf.loc[bestE_idx, 'coreF']
        bestM = cur_perf.loc[bestE_idx, 'memF']
        predictE = min(modelledE)
        predictE_idx = np.argmin(modelledE)
        predictC = cur_perf.loc[predictE_idx, 'coreF']
        predictM = cur_perf.loc[predictE_idx, 'memF']

        item['bestE'] = bestE
        item['bestC'] = bestC
        item['bestM'] = bestM
        item['predictE'] = predictE
        item['predictC'] = predictC
        item['predictM'] = predictM


    print energy_data

if __name__ == '__main__':

    if not os.path.exists("figures"):
        os.makedirs("figures")

    # gpu card and data file
    method = 'qiang2018'
    ml_algo = 'svr-poly'

    ## pipeline paper, plot error heatmap of different frequency settings
    gpu = 'gtx980-low-dvfs'
    version = 'real-small-workload'
    #plot_perf_acc_freq(gpu, version, method, save_filename='%s-%s-%s-acc-dvfs' % (gpu, version, method))
    #gpu = 'gtx980-high-dvfs'
    #version = 'real-small-workload'
    #plot_perf_acc_freq(gpu, version, method, save_filename='%s-%s-%s-acc-dvfs' % (gpu, version, method))
    ##gpu = 'gtx1080ti-dvfs'
    ##version = 'real'
    ##plot_perf_acc_freq(gpu, version, method, save_filename='%s-%s-%s-acc-dvfs' % (gpu, version, method))
    #gpu = 'p100-dvfs'
    #version = 'real'
    #plot_perf_acc_freq(gpu, version, method, save_filename='%s-%s-%s-acc-dvfs' % (gpu, version, method))
    #plot_perf_acc_freq_merge(save_filename='acc-freq-merge')
    plot_energy(gpu, version, save_filename='%s-%s-%s-energy' % (gpu, version, method))

    ## pipeline paper, plot performance scaling behavior in motivation part
    #csv_file = "csvs/raw/%s-%s-Performance.csv" % (gpu, version)
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

    #gpu = 'gtx980'
    #plot_perf_acc_dvfs(gpu, 'qiang2018', '%s_analytical' % gpu)
    #gpu = 'gtx1080ti'
    #plot_perf_acc_dvfs(gpu, 'qiang2018', '%s_analytical' % gpu)
    #gpu = 'p100'
    #plot_perf_acc_dvfs(gpu, 'qiang2018', '%s_analytical' % gpu)

    #gpu = 'gtx980-high-dvfs'
    #version = 'real-small-workload'
    #plot_perf_acc_analytical(gpu, version, method, '%s_analytical' % gpu)
    #gpu = 'gtx980-low-dvfs'
    #version = 'real-small-workload'
    #plot_perf_acc_analytical(gpu, version, method, '%s_analytical' % gpu)
    #gpu = 'gtx1080ti-dvfs'
    #version = 'real'
    #plot_perf_acc_analytical(gpu, version, method, '%s_analytical' % gpu)
    #gpu = 'p100-dvfs'
    #version = 'real'
    #plot_perf_acc_analytical(gpu, version, method, '%s_analytical' % gpu)

    # pipeline paper, plot err and correlation scatter
    method = 'qiang2018'
    gpu = 'gtx980-low-dvfs'
    version = 'real-small-workload'
    plot_perf_acc_corr(gpu, version, method, '%s_%s_%s_err_corr' % (gpu, version, method))
    #gpu = 'gtx980-high-dvfs'
    #version = 'real-small-workload'
    #plot_perf_acc_corr(gpu, version, method, '%s_%s_%s_err_corr' % (gpu, version, method))
    gpu = 'gtx1080ti-dvfs'
    version = 'real'
    plot_perf_acc_corr(gpu, version, method, '%s_%s_%s_err_corr' % (gpu, version, method))
    gpu = 'p100-dvfs'
    version = 'real'
    plot_perf_acc_corr(gpu, version, method, '%s_%s_%s_err_corr' % (gpu, version, method))

    ## pipeline paper, plot dvfs-roofline model
    #gpu = 'gtx980-low-dvfs'
    #version = 'real-small-workload'
    #csv_file = "csvs/analytical/cycles/%s-%s-qiang2018-cycles.csv" % (gpu, version)
    #df = pd.read_csv(csv_file, header = 0)
    #kernels = list(df['appName'].drop_duplicates())
    #print kernels
    #
    #kernels = ['histogram', 'BlackScholes', 'backpropForward']
    #for kernel in kernels:
    #    print kernel
    #    plot_dvfs_roofline(gpu, version, kernel, save_filename = '%s_%s_%s_dvfs_roofline' % (gpu, version, kernel))

    ## pipeline paper, plot dvfs scaling effect and instruction distributions
    #gpu = 'gtx980-low-dvfs'
    #version = 'real-small-workload'

    #csv_file = "csvs/v1/%s-%s-Performance.csv" % (gpu, version)
    #plot_dvfs_scaling(gpu, csv_file)
    #plot_inst_distribution(gpu, csv_file, 'gtx980_sample_inst_dist')

