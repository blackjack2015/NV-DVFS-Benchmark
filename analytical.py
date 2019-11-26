import pandas as pd
import numpy as np
import sys, os
from settings import *
import argparse
import math

if not os.path.exists("csvs/analytical/cycles"):
    os.makedirs("csvs/analytical/cycles")
if not os.path.exists("csvs/analytical/features"):
    os.makedirs("csvs/analytical/features")
if not os.path.exists("csvs/analytical/results"):
    os.makedirs("csvs/analytical/results")

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, help='data file path', default='raw')
parser.add_argument('--benchmark-setting', type=str, help='gpu and dvfs setting', default='gtx980-low-dvfs')
parser.add_argument('--kernel-setting', type=str, help='kernel list', default='real-small-workload')
parser.add_argument('--method', type=str, help='analytical modeling method', default='qiang2018')

opt = parser.parse_args()
print opt

lowest_core = 500
lowest_mem = 500

gpucard = opt.benchmark_setting
kernel_setting = opt.kernel_setting
method = opt.method
data_root = opt.data_root

csv_perf = "csvs/%s/%s-%s-Performance.csv" % (data_root, gpucard, kernel_setting)
df = pd.read_csv(csv_perf, header = 0)

if 'gtx980' in gpucard:
    GPUCONF = GTX980()
if 'gtx1080ti'in gpucard:
    GPUCONF = GTX1080TI()
if 'titanx' in gpucard:
    GPUCONF = TITANX()
if 'p100' in gpucard:
    GPUCONF = P100()
if 'v100' in gpucard:
    GPUCONF = V100()

# experimental test
#pointer = ['convolutionTexture', 'nn', 'SobolQRNG', 'reduction', 'hotspot'] 
pointer = []
extras = ['backpropBackward', 'binomialOptions', 'cfd', 'eigenvalues', 'gaussian', 'srad', 'dxtc', 'pathfinder', 'scanUniformUpdate', 'stereoDisparity'] 
#extras = []
#extras += ['quasirandomGenerator', 'matrixMulGlobal', 'mergeSort']
#extras += ['histogram', 'matrixMulGlobal', 'mergeSort', 'quasirandomGenerator']
df = df[~df.appName.isin(extras) & ~df.appName.isin(pointer) & (df.coreF>=lowest_core) & (df.memF>=lowest_mem)]
#df = df[~df.appName.isin(extras) & (df.coreF>=lowest_core) & (df.memF>=lowest_mem)]
df = df.reset_index(drop=True)
df = df.sort_values(by = ['appName', 'coreF', 'memF'])

features = pd.DataFrame(columns=['appName', 'coreF', 'memF', 'n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'l2_miss', 'l2_hit', 'mem_insts', 'insts', 'act_util', 'L_DM', 'D_DM']) # real_cycle per round
features['appName'] = df['appName']
features['coreF'] = df['coreF']
features['memF'] = df['memF']
# shared memory information
features['n_shm_ld'] = df['shared_load_transactions'] / df['warps']
features['n_shm_st'] = df['shared_store_transactions'] / df['warps']
features['n_shm'] = features['n_shm_ld'] + features['n_shm_st']

# global memory information
#try:
#    features['n_gld'] = df['gld_transactions'] / df['warps']
#    features['n_gst'] = df['gst_transactions'] / df['warps']
#except Exception as e:
#    features['n_gld'] = df['l2_read_transactions'] / df['warps']
#    features['n_gst'] = df['l2_write_transactions'] / df['warps']

features['n_gld'] = df['l2_read_transactions'] / df['warps']
features['n_gst'] = df['l2_write_transactions'] / df['warps']
try:
    features['tex_trans'] = df['tex_cache_transactions'] / df['warps'] 
    features.loc[features['tex_trans'] < 0, 'tex_trans'] = 0
except Exception as e:
    features['tex_trans'] = 0

#features['n_gld'] = (df['l2_read_transactions'] + df['shared_load_transactions']) / df['warps']
#features['n_gst'] = (df['l2_write_transactions'] + df['shared_store_transactions']) / df['warps']

# l2 information
#features['l2_miss'] = df['dram_read_transactions'] / df['l2_read_transactions']
#features['l2_miss'] = df['dram_write_transactions'] / df['l2_write_transactions']
#features['l2_miss'] = (df['dram_read_transactions'] + df['dram_write_transactions']) / (df['l2_read_transactions'] + df['l2_write_transactions'])
features['l2_miss'] = (df['dram_read_transactions'] + df['dram_write_transactions']) / ((features['n_gst'] + features['n_gld']) * df['warps'])
features.loc[features['l2_miss'] > 1, 'l2_miss'] = 1
features['l2_hit'] = 1 - features['l2_miss']

# compute instructions
try:
    features['fp_insts'] = df['inst_fp_32'] / (df['warps'] * 32.0)
    features['dp_insts'] = df['inst_fp_64'] / (df['warps'] * 32.0)
    #features['int_insts'] = df['inst_integer'] / (df['warps'] * 32.0)
    #features['insts'] = features['fp_insts'] + features['dp_insts'] * 2.0 + features['int_insts']
except Exception as e:
    print "No float/double instruction information..."

try:
    features['int_insts'] = df['inst_integer'] / (df['warps'] * 32.0)
    #features['insts'] = features['fp_insts'] + features['dp_insts'] * 2.0 + features['int_insts']
except Exception as e:
    print "No integer instruction information..."

features['mem_insts'] = features['n_gld'] + features['n_gst'] + features['n_shm_ld'] + features['n_shm_st'] / 4.0
features['insts'] = df['inst_per_warp'] - features['mem_insts'] # + features['dp_insts'] * 3.0
features['branch_insts'] = df['cf_executed'] / (df['warps'] * 32.0)

# other parameters
features.loc[features['insts'] < 0, 'insts'] = 0
features['act_util'] = df['achieved_occupancy']
features['L_DM'] = GPUCONF.a_L_DM * df['coreF'] / df['memF'] + GPUCONF.b_L_DM
features['D_DM'] = (GPUCONF.a_D_DM / df['memF'] + GPUCONF.b_D_DM) * df['coreF'] / df['memF']

# add bias to model parameters
#features['L_DM'] = features['L_DM'] * 1.2
#features['D_DM'] = features['D_DM'] * 1.2
#features['act_util'] = features['act_util'] * 1.2
#features['l2_hit'] = features['l2_hit'] * 0.8 

# remove shm part if hong2009
if method == 'hong2009':
    filter_out_shm = features.n_shm == 0

    features = features[filter_out_shm]
    features = features.reset_index(drop=True)

    df = df[filter_out_shm]
    df = df.reset_index(drop=True)

# save featuress to csv/xlsx
features.to_csv("csvs/analytical/features/%s-%s-features.csv" % (gpucard, kernel_setting))
#writer = pd.ExcelWriter("csvs/analytical/features/%s-%s-features.xlsx" % (gpucard, kernel_setting))
#features.to_excel(writer, 'Sheet1')
#writer.save()

# other methodology
def hong2009(df):

    # analytical model
    cycles = pd.DataFrame(columns=['appName', 'coreF', 'memF', 'cold_miss', 'c_to_m', 'modelled_cycle', 'real_cycle', 'c1', 'c2', 'c3']) # real_cycle per round
    cycles['appName'] = df['appName']
    cycles['coreF'] = df['coreF']
    cycles['memF'] = df['memF']
    cycles['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    cycles['cold_miss'] = df['L_DM']

    cycles['depart_delay'] = df['D_DM'] * df['l2_miss'] + GPUCONF.D_L2 * df['l2_hit']
    cycles['mem_l'] = df['L_DM'] * df['l2_miss'] + GPUCONF.L_L2 * df['l2_hit']
    cycles['N'] = GPUCONF.WARPS_MAX * df['act_util']
    cycles['compute_cycles'] = df['insts'] * GPUCONF.D_INST
    cycles['compute_cycles_per_period'] = cycles['compute_cycles'] / (df['n_gld'] + df['n_gst']) #* GPUCONF.D_INST
    cycles['mem_cycles'] = cycles['depart_delay'] * (df['n_gld'] + df['n_gst']) 
    cycles['MWP_without_BW'] = cycles['mem_l'] / cycles['depart_delay'] 
    cycles['MWP_peak_BW'] = cycles['mem_l'] / GPUCONF.SM_COUNT
    #cycles['MWP'] = cycles[['MWP_without_BW','MWP_peak_BW', 'N']].min(axis=1)
    cycles['MWP'] = cycles['MWP_without_BW']
    cycles['CWP_without_OCC'] = (cycles['compute_cycles'] + cycles['mem_cycles']) / cycles['compute_cycles']
    cycles['CWP'] = cycles[['CWP_without_OCC', 'N']].min(axis=1)
    
    cycles['shm_insts'] = df['n_shm_ld'] + df['n_shm_st']
    #cycles['mem_cycles'] = (df['n_gld'] + df['n_gst']) * cycles['mem_l'] * (GPUCONF.WARPS_MAX * df['act_util'] /cycles['MWP'])
    #cycles['compute_cycles'] = df['insts'] * GPUCONF.D_INST

    for idx, item in cycles.iterrows():

        cycles.loc[idx, 'c1'] = cycles.loc[idx, 'mem_cycles'] + cycles.loc[idx, 'compute_cycles'] + cycles.loc[idx, 'compute_cycles'] / (df.loc[idx, 'n_gld'] + df.loc[idx, 'n_gst']) * (cycles.loc[idx, 'MWP'] - 1)
       
        cycles.loc[idx, 'c2'] = cycles.loc[idx, 'mem_cycles'] * cycles.loc[idx, 'N'] / cycles.loc[idx, 'MWP'] + cycles.loc[idx, 'compute_cycles'] / (df.loc[idx, 'n_gld'] + df.loc[idx, 'n_gst']) * (cycles.loc[idx, 'MWP'] - 1) 

        cycles.loc[idx, 'c3'] = cycles.loc[idx, 'mem_l'] + cycles.loc[idx, 'compute_cycles'] * cycles.loc[idx, 'N']

        if cycles.loc[idx, 'MWP'] == cycles.loc[idx, 'N'] and cycles.loc[idx, 'MWP'] == cycles.loc[idx, 'N']:  # not enough warp
            cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'c1'] 
        elif cycles.loc[idx, 'CWP'] >= cycles.loc[idx, 'MWP']:  # memory bound
            cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'c2'] 
        elif cycles.loc[idx, 'MWP'] >= cycles.loc[idx, 'CWP'] or cycles.loc[idx, 'compute_cycles'] > cycles.loc[idx, 'mem_cycles']:  # compute bound
            cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'c3'] 

        #compute_bound = cycles.loc[idx, 'compute_cycles'] * GPUCONF.WARPS_MAX * df.loc[idx, 'act_util'] + cycles.loc[idx, 'mem_l']
        #memory_bound = cycles.loc[idx, 'mem_cycles'] + cycles.loc[idx, 'compute_cycles']
        #if compute_bound > memory_bound:
        #    cycles.loc[idx, 'modelled_cycle'] = compute_bound
        #else:
	#    cycles.loc[idx, 'modelled_cycle'] = memory_bound

        #if df.loc[idx, 'act_util'] <= 0.38:
        #    cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'mem_cycles'] + cycles.loc[idx, 'compute_cycles'] 

    cycles = cycles.sort_values(by=['appName', 'c_to_m'])
    return cycles


def song2013(df):

    # analytical model
    cycles = pd.DataFrame(columns=['appName', 'coreF', 'memF', 'cold_miss', 'c_to_m', 'modelled_cycle', 'real_cycle']) # real_cycle per round
    cycles['appName'] = df['appName']
    cycles['coreF'] = df['coreF']
    cycles['memF'] = df['memF']
    cycles['c_to_m'] = df['coreF'] * 1.0 / df['memF']

    cycles['depart_delay'] = df['D_DM'] * df['l2_miss'] + GPUCONF.D_L2 * df['l2_hit']
    cycles['mem_l'] = df['L_DM'] * df['l2_miss'] + GPUCONF.L_L2 * df['l2_hit']
    # global load and store
    cycles['g_load'] = cycles['mem_l'] + (df['n_gld'] - 1) * cycles['depart_delay']
    cycles['g_store'] = cycles['mem_l'] + (df['n_gst'] - 1) * cycles['depart_delay']

    # sync
    cycles['sync'] = (df['act_util'] * GPUCONF.WARPS_MAX - 1) * cycles['depart_delay']

    # compute
    cycles['compute'] = GPUCONF.D_INST * 32.0 / GPUCONF.CORES_SM * df['insts']

    # shared memory
    cycles['shared'] = GPUCONF.D_sh * (df['n_shm_ld'] + df['n_shm_st']) * df['act_util'] * GPUCONF.WARPS_MAX 

    cycles['modelled_cycle'] = cycles['compute'] + cycles['g_load'] + cycles['g_store'] + cycles['compute'] * (df['act_util'] * GPUCONF.WARPS_MAX - 1) + cycles['shared'] + cycles['sync']
    cycles = cycles.sort_values(by=['appName', 'c_to_m'])
    return cycles

def qiang2018(df):

    # analytical model
    cycles = pd.DataFrame(columns=['appName', 'coreF', 'memF', 'cold_miss', 'c_to_m', 'modelled_cycle', 'real_cycle']) # real_cycle per round
    cycles['appName'] = df['appName']
    cycles['coreF'] = df['coreF']
    cycles['memF'] = df['memF']
    cycles['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    cycles['cold_miss'] = df['L_DM']
    cycles['avg_mem_lat'] = ((df['L_DM'] + df['D_DM']) * (1 - df['l2_hit']) + GPUCONF.L_L2 * df['l2_hit']) 
    cycles['avg_mem_del'] = (df['D_DM'] * (1 - df['l2_hit']) + GPUCONF.D_L2 * df['l2_hit']) 
    cycles['mem_del'] = (df['n_gld'] + df['n_gst']) * cycles['avg_mem_del'] * GPUCONF.WARPS_MAX * df['act_util'] # memory queue delay for all warps per round
    cycles['mem_lat'] = (df['n_gld'] + df['n_gst']) * cycles['avg_mem_lat'] / 4.0 # memory latency for one warp per round
    cycles['shm_del'] = GPUCONF.D_sh * (df['n_shm_ld'] + df['n_shm_st']) * df['act_util'] * GPUCONF.WARPS_MAX + GPUCONF.L_sh # shared queue delay for all warps per round
    cycles['tex_del'] = df['tex_trans'] * df['act_util'] * GPUCONF.WARPS_MAX / GPUCONF.TEX_UNITS * GPUCONF.D_TEX
    cycles['dp_del'] = df['dp_insts'] * df['act_util'] * GPUCONF.WARPS_MAX * GPUCONF.D_DP
    #cycles['int_del'] = df['int_insts'] * df['act_util'] * GPUCONF.WARPS_MAX
    cycles['branch_del'] = df['branch_insts'] * df['act_util'] * GPUCONF.WARPS_MAX * 32.0 * 0.5
    #cycles['tex_del'] = 0
    cycles['shm_offset'] = ((df['n_shm_ld'] + df['n_shm_st']) * 1.0 / (df['n_gld'] + df['n_gst'])) * GPUCONF.L_sh
    cycles['shm_lat'] = (df['n_shm_ld'] + df['n_shm_st']) * GPUCONF.L_sh # shared latency for one warp per round
    cycles['compute_del'] = GPUCONF.D_INST * (df['insts']) * df['act_util'] * 32.0 * GPUCONF.WARPS_MAX / GPUCONF.CORES_SM + GPUCONF.L_INST # compute delay for all warps per round
    cycles['compute_offset'] = df['insts'] * 1.0 / (df['n_gld'] + df['n_gst']) * GPUCONF.L_INST
    cycles['compute_lat'] = df['insts'] * GPUCONF.L_INST # compute latency for one warp per round
    cycles['sm_del'] = (cycles['compute_del'] + cycles['shm_del'] + cycles['dp_del']) 
    #cycles['sm_del'] = (cycles['compute_del'] + cycles['shm_del']) 
    cycles['sm_lat'] = cycles['compute_lat'] + cycles['shm_lat']
    #cycles['sm_op'] = df['insts'] * L_INST
    cycles['insts'] = df['insts']
    
    # add type for offset
    #cycles['offset'] = None

    lack_thres = 0.25 # for p100, 0.3 gives in marginally better results. 
    for idx, item in cycles.iterrows():
        # app using texture memory
        if item.appName == 'convolutionTexture':
            cycles.loc[idx, 'mem_del'] += cycles.loc[idx, 'tex_del']
        # app using many branch instructions, strange for v100
        if (item.appName in ['reduction']) and (not "v100" in gpucard):
            cycles.loc[idx, 'sm_del'] += cycles.loc[idx, 'branch_del'] 
        # app only have dram write transactions
        #if item.appName == 'quasirandomGenerator':
        #    cycles.loc[idx, 'mem_del'] = df.loc[idx, 'n_gst'] * df.loc[idx, 'D_DM'] * GPUCONF.WARPS_MAX * df.loc[idx, 'act_util'] * 1.1

        if cycles.loc[idx, 'sm_del'] > cycles.loc[idx, 'mem_del']:
            cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'sm_del'] #+ cycles.loc[idx, 'avg_mem_lat']
        else:
	    cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'mem_del'] #+ cycles.loc[idx, 'avg_mem_lat']

        if (item.appName != 'nn') or (cycles.loc[idx, 'modelled_cycle'] < 2800):
            cycles.loc[idx, 'modelled_cycle'] += cycles.loc[idx, 'cold_miss'] 

        #if df.loc[idx, 'act_util'] <= 0.30:
        #    if cycles.loc[idx, 'sm_del'] + cycles.loc[idx, 'mem_lat'] > cycles.loc[idx, 'sm_lat'] + cycles.loc[idx, 'mem_del']:
        #        cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'sm_del'] + cycles.loc[idx, 'mem_lat']
        #    else:
        #        cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'sm_lat'] + cycles.loc[idx, 'mem_del']

        #special = ['hotspot', 'convolutionTexture', 'nn']
        if df.loc[idx, 'act_util'] <= lack_thres:
        #if df.loc[idx, 'appName'] in special:
            lack_wait = 0.5 * cycles.loc[idx, 'avg_mem_lat'] + cycles.loc[idx, 'compute_offset'] + cycles.loc[idx, 'avg_mem_del'] * GPUCONF.WARPS_MAX * df.loc[idx, 'act_util'] + 0.5 * cycles.loc[idx, 'avg_mem_lat'] + (cycles.loc[idx, 'compute_offset'] + cycles.loc[idx, 'avg_mem_lat']) * (df.loc[idx, 'n_gld'] + df.loc[idx, 'n_gst'] - 1) / 4.0
            lack_no_wait = cycles.loc[idx, 'compute_offset'] * (GPUCONF.WARPS_MAX * df.loc[idx, 'act_util'] - 1) + (cycles.loc[idx, 'compute_offset'] + cycles.loc[idx, 'avg_mem_lat']) * (df.loc[idx, 'n_gld'] + df.loc[idx, 'n_gst']) / 4.0
            if lack_wait > lack_no_wait:
                cycles.loc[idx, 'modelled_cycle'] = lack_wait
            else:
                cycles.loc[idx, 'modelled_cycle'] = lack_no_wait


    cycles = cycles.sort_values(by=['appName', 'c_to_m'])
    #for idx, item in df.iterrows():
    	#cur_name = df['appName'][idx]
    	#if GPUCONF.eqType[cur_name] == DM_HID:
    	#	cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'cold_miss']
    	#elif GPUCONF.eqType[cur_name] == COMP_HID:
    	#	cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'sm_op']
    	#elif GPUCONF.eqType[cur_name] == MEM_HID:
    	#	cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'mem_op']
    	#elif GPUCONF.eqType[cur_name] == DM_COMP_HID:
    	#	cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'cold_miss'] -cycles.loc[idx, 'sm_op']
    	#elif GPUCONF.eqType[cur_name] == MEM_LAT_BOUND:
    	#	cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'mem_op'] -cycles.loc[idx, 'cold_miss'] +cycles.loc[idx, 'lat_op']
    	#elif GPUCONF.eqType[cur_name] == NO_HID:
    	#	cycles.loc[idx, 'offset'] = 0
    	#elif GPUCONF.eqType[cur_name] == MIX:
    	#	cycles.loc[idx, 'offset'] = 0
    	#elif GPUCONF.eqType[cur_name] == COMP_BOUND:
    	#	cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'sm_op'] - cycles.loc[idx, 'mem_op'] + cycles.loc[idx, 'compute_del']
    	#elif GPUCONF.eqType[cur_name] == SHM_BOUND:
    	#	cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'sm_op'] - cycles.loc[idx, 'mem_op'] + cycles.loc[idx, 'shm_del']
    	#else:
    	#	print "Invalid modeling type of %s..." % cur_name
    	#	sys.exit(-1)	
    
    #cycles['modelled_cycle'] = cycles['cold_miss'] + cycles['mem_op'] + cycles['sm_op'] + cycles['offset']

    return cycles

if method == 'qiang2018':
    cycles = qiang2018(features)
elif method == 'song2013':
    cycles = song2013(features)
elif method == 'hong2009':
    cycles = hong2009(features)

def print_kernel(cycles, kernel):
    kernel_idx = cycles.appName == kernel
    print cycles[kernel_idx][['real_cycle', 'modelled_cycle', 'mem_del', 'sm_del', 'tex_del', 'branch_del']]

cycles['exec_rounds'] = df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy'])
#cycles['exec_rounds'] = cycles['exec_rounds'].astype(int)
cycles['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / cycles['exec_rounds']
cycles['abe'] = abs(cycles['modelled_cycle'] - cycles['real_cycle']) / cycles['real_cycle']
#print_kernel(cycles, 'quasirandomGenerator')

# save results to csv/xlsx
cycles.to_csv("csvs/analytical/cycles/%s-%s-%s-cycles.csv" % (gpucard, kernel_setting, method))
#writer = pd.ExcelWriter("csvs/analytical/cycles/%s-%s-%s-cycles.xlsx" % (gpucard, kernel_setting, method))
#cycles.to_excel(writer, 'Sheet1')
#writer.save()

kernels = features['appName'].drop_duplicates()
kernels.sort_values(inplace=True)

f = open("csvs/analytical/results/%s-%s-%s-dvfs.csv" % (gpucard, kernel_setting, method), "w")
f.write("kernel,coreF,memF,real,predict,error\n")
for idx, item in cycles.iterrows():
	kernel = item['appName']
        coreF = item['coreF']
        memF = item['memF']
        real = item['real_cycle']
        predict = item['modelled_cycle']
        error = abs(item['real_cycle'] - item['modelled_cycle']) / item['real_cycle']

        f.write("%s,%d,%d,%f,%f,%f\n" % (kernel, coreF, memF, real, predict, error))
f.close()


f = open("csvs/analytical/results/%s-%s-%s-aver.csv" % (gpucard, kernel_setting, method), "w")
f.write("kernel,ape\n")
for kernel in kernels:
	tmp_cycles = cycles.loc[df['appName'] == kernel]
	tmp_ape = np.mean(tmp_cycles['abe'])
	tmp_err_std = np.std(tmp_cycles['abe'])

	print "%s:%f, %f" % (kernel, tmp_ape, tmp_err_std)
	f.write("%s,%f\n" % (kernel, tmp_ape))
f.close()


errors = []
for i in range(len(cycles['modelled_cycle'])):
        if cycles['appName'][i] in pointer or cycles['appName'][i] in extras:
            continue
 
        #if cycles['coreF'][i] >= 500 and cycles['memF'][i] >= 500:
	errors.append(cycles['abe'][i])

    	    #print cycles['appName'][i], cycles['coreF'][i], cycles['memF'][i]
    	    ##print i, df['appName'][i], 'relative error', cycles['abe'][i]
    	    ## print i, df['appName'][i]
    	    #print 'n_gld', features['n_gld'][i]
    	    #print 'n_gst', features['n_gst'][i]
    	    #print 'l2_hit', features['l2_hit'][i]
    	    #print 'n_shm_ld', features['n_shm_ld'][i]
    	    #print 'n_shm_st', features['n_shm_st'][i]
    	    #print 'insts', features['insts'][i]
    	    #print 'act_util', features['act_util'][i]
    	    #print 'dram delay', features['D_DM'][i]
    	    #print 'dram latency', features['L_DM'][i]
    	    #print 'coreF', features['coreF'][i]
    	    #print 'memF', features['memF'][i]
    	    #print 'mem_del', cycles['mem_del'][i]
    	    #print 'sm_del', cycles['sm_del'][i]
    	    #print 'modelled', cycles['modelled_cycle'][i]
    	    #print 'real', cycles['real_cycle'][i], df['time/ms'][i], "ms"
    	    #print 'relative error', cycles['abe'][i]
    	    #print '\n'
 
pos_50 = int(len(errors) * 0.50)
pos_75 = int(len(errors) * 0.75)
pos_95 = int(len(errors) * 0.95)
errors = np.sort(errors)
print "50th percentile:", errors[pos_50]
print "75th percentile:", errors[pos_75]
print "95th percentile:", errors[pos_95]
print "MAPE of %d samples: %f" % (len(errors), np.mean(errors))
#if 'gtx980' in gpucard:
#    print "sensitive error:", (np.mean(errors) - 0.03854) * 100
#if 'gtx1080ti'in gpucard:
#    print "sensitive error:", (np.mean(errors) - 0.08596) * 100
#if 'p100' in gpucard:
#    print "sensitive error:", (np.mean(errors) - 0.10378) * 100
