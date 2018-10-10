import pandas as pd
import numpy as np
import sys
from settings import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark-setting', type=str, help='gpu and dvfs setting', default='gtx980-DVFS')
parser.add_argument('--kernel-setting', type=str, help='kernel list', default='real')
parser.add_argument('--method', type=str, help='analytical modeling method', default='qiang2018')

opt = parser.parse_args()
print opt

gpucard = opt.benchmark_setting
kernel_setting = opt.kernel_setting
method = opt.method
#csv_perf = "csvs/%s-%s-Performance.csv" % (gpucard, kernel_setting)
csv_perf = "csvs/v0/%s-%s-Performance.csv" % (gpucard, kernel_setting)
#csv_perf = "csvs/v1/%s-%s-Performance.csv" % (gpucard, kernel_setting)
df = pd.read_csv(csv_perf, header = 0)

if 'gtx980' in gpucard:
    GPUCONF = GTX980()
if 'gtx1080ti'in gpucard:
    GPUCONF = GTX1080TI()
if 'titanx' in gpucard:
    GPUCONF = TITANX()
if 'p100' in gpucard:
    GPUCONF = P100()

# experimental test
pointer = ['convolutionTexture', 'nn', 'SobolQRNG', 'reduction', 'hotspot'] 
#pointer = []
extras = ['backpropBackward', 'binomialOptions', 'cfd', 'eigenvalues', 'gaussian', 'srad', 'dxtc', 'pathfinder', 'scanUniformUpdate', 'stereoDisparity'] 
#extras = []

features = pd.DataFrame(columns=['appName', 'coreF', 'memF', 'n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'l2_miss', 'l2_hit', 'mem_insts', 'insts', 'act_util', 'L_DM', 'D_DM']) # real_cycle per round
features['appName'] = df['appName']
features['coreF'] = df['coreF']
features['memF'] = df['memF']
# shared memory information
features['n_shm_ld'] = df['shared_load_transactions'] / df['warps']
features['n_shm_st'] = df['shared_store_transactions'] / df['warps']

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
#features['fp_insts'] = df['inst_fp_32'] / (df['warps'] * 32.0)
#features['dp_insts'] = df['inst_fp_64'] / (df['warps'] * 32.0)
#features['int_insts'] = df['inst_integer'] / (df['warps'] * 32.0)
#features['insts'] = features['fp_insts'] + features['dp_insts'] * 2.0 + features['int_insts']
features['mem_insts'] = features['n_gld'] + features['n_gst'] + features['n_shm_ld'] + features['n_shm_st'] / 4.0
features['insts'] = df['inst_per_warp'] - features['mem_insts'] # + features['dp_insts']

# other parameters
features.loc[features['insts'] < 0, 'insts'] = 0
features['act_util'] = df['achieved_occupancy']
features['L_DM'] = GPUCONF.a_L_DM * df['coreF'] / df['memF'] + GPUCONF.b_L_DM
features['D_DM'] = (GPUCONF.a_D_DM / df['memF'] + GPUCONF.b_D_DM) * df['coreF'] / df['memF']

# save featuress to csv/xlsx
features.to_csv("csvs/analytical/features/%s-%s-features.csv" % (gpucard, kernel_setting))
writer = pd.ExcelWriter("csvs/analytical/features/%s-%s-features.xlsx" % (gpucard, kernel_setting))
features.to_excel(writer, 'Sheet1')
writer.save()

# other methodology
def hong2009(df):

    # analytical model
    cycles = pd.DataFrame(columns=['appName', 'coreF', 'memF', 'cold_miss', 'c_to_m', 'modelled_cycle', 'real_cycle']) # real_cycle per round
    cycles['appName'] = df['appName']
    cycles['coreF'] = df['coreF']
    cycles['memF'] = df['memF']
    cycles['c_to_m'] = df['coreF'] * 1.0 / df['memF']
    cycles['cold_miss'] = df['L_DM']

    cycles['depart_delay'] = df['D_DM']
    cycles['mem_l'] = df['L_DM']
    cycles['MWP'] = cycles['mem_l'] / cycles['depart_delay']
    cycles['mem_cycles'] = (df['n_gld'] + df['n_gst']) * cycles['mem_l'] * (GPUCONF.WARPS_MAX * df['act_util'] /cycles['MWP'])
    cycles['compute_cycles'] = df['insts'] * GPUCONF.D_INST


    for idx, item in cycles.iterrows():

        compute_bound = cycles.loc[idx, 'compute_cycles'] * GPUCONF.WARPS_MAX * df.loc[idx, 'act_util'] + cycles.loc[idx, 'mem_l']
        memory_bound = cycles.loc[idx, 'mem_cycles'] + cycles.loc[idx, 'compute_cycles']
        if compute_bound > memory_bound:
            cycles.loc[idx, 'modelled_cycle'] = compute_bound
        else:
	    cycles.loc[idx, 'modelled_cycle'] = memory_bound

        if df.loc[idx, 'act_util'] <= 0.38:
            cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'mem_cycles'] + cycles.loc[idx, 'compute_cycles'] 

    cycles = cycles.sort_values(by=['appName', 'c_to_m'])
    return cycles


def song2013(df):

    # analytical model
    cycles = pd.DataFrame(columns=['appName', 'coreF', 'memF', 'cold_miss', 'c_to_m', 'modelled_cycle', 'real_cycle']) # real_cycle per round
    cycles['appName'] = df['appName']
    cycles['coreF'] = df['coreF']
    cycles['memF'] = df['memF']
    cycles['c_to_m'] = df['coreF'] * 1.0 / df['memF']

    # global load and store
    cycles['g_load'] = df['L_DM'] + (df['n_gld'] - 1) * df['D_DM']
    cycles['g_store'] = df['L_DM'] + (df['n_gst'] - 1) * df['D_DM']

    # sync
    cycles['sync'] = (df['act_util'] * GPUCONF.WARPS_MAX - 1) * df['D_DM']

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
    cycles['mem_del'] = (df['n_gld'] + df['n_gst']) * (df['D_DM'] * (1 - df['l2_hit']) + GPUCONF.D_L2 * df['l2_hit']) * GPUCONF.WARPS_MAX * df['act_util'] # memory queue delay for all warps per round
    cycles['mem_lat'] = (df['n_gld'] + df['n_gst']) * ((df['L_DM'] + df['D_DM']) * (1 - df['l2_hit']) + GPUCONF.L_L2 * df['l2_hit']) / 4.0 # memory latency for one warp per round
    cycles['shm_del'] = GPUCONF.D_sh * (df['n_shm_ld'] + df['n_shm_st']) * df['act_util'] * GPUCONF.WARPS_MAX + GPUCONF.L_sh # shared queue delay for all warps per round
    cycles['shm_offset'] = ((df['n_shm_ld'] + df['n_shm_st']) * 1.0 / (df['n_gld'] + df['n_gst'])) * GPUCONF.L_sh
    cycles['shm_lat'] = (df['n_shm_ld'] + df['n_shm_st']) * GPUCONF.L_sh # shared latency for one warp per round
    cycles['compute_del'] = GPUCONF.D_INST * df['insts'] * df['act_util'] * 32.0 * GPUCONF.WARPS_MAX / GPUCONF.CORES_SM + GPUCONF.L_INST # compute delay for all warps per round
    cycles['compute_offset'] = df['insts'] * 1.0 / (df['n_gld'] + df['n_gst']) * GPUCONF.L_INST
    cycles['compute_lat'] = df['insts'] * GPUCONF.L_INST # compute latency for one warp per round
    cycles['sm_del'] = (cycles['compute_del'] + cycles['shm_del']) 
    cycles['sm_lat'] = cycles['compute_lat'] + cycles['shm_lat']
    cycles['tex_del'] = df['tex_trans'] * GPUCONF.WARPS_MAX * df['act_util']
    #cycles['sm_op'] = df['insts'] * L_INST
    cycles['insts'] = df['insts']
    
    # add type for offset
    #cycles['offset'] = None

    for idx, item in cycles.iterrows():
        if cycles.loc[idx, 'sm_del'] > cycles.loc[idx, 'mem_del']:
            cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'sm_del'] 
        else:
	    cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'mem_del'] 

        if df.loc[idx, 'act_util'] <= 0.38:
            if cycles.loc[idx, 'sm_del'] + cycles.loc[idx, 'mem_lat'] > cycles.loc[idx, 'sm_lat'] + cycles.loc[idx, 'mem_del']:
                cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'sm_del'] + cycles.loc[idx, 'mem_lat']
            else:
                cycles.loc[idx, 'modelled_cycle'] = cycles.loc[idx, 'sm_lat'] + cycles.loc[idx, 'mem_del']


    cycles['modelled_cycle'] += cycles['cold_miss'] 
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

cycles['exec_rounds'] = df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy'])
#cycles['exec_rounds'] = cycles['exec_rounds'].astype(int)
cycles['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / cycles['exec_rounds']
cycles['abe'] = abs(cycles['modelled_cycle'] - cycles['real_cycle']) / cycles['real_cycle']

# save results to csv/xlsx
cycles.to_csv("csvs/analytical/cycles/%s-%s-%s-cycles.csv" % (gpucard, kernel_setting, method))
writer = pd.ExcelWriter("csvs/analytical/cycles/%s-%s-%s-cycles.xlsx" % (gpucard, kernel_setting, method))
cycles.to_excel(writer, 'Sheet1')
writer.save()

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

	if kernel in pointer or kernel in extras:
	    continue
        f.write("%s,%d,%d,%f,%f,%f\n" % (kernel, coreF, memF, real, predict, error))
f.close()


f = open("csvs/analytical/results/%s-%s-%s-aver.csv" % (gpucard, kernel_setting, method), "w")
f.write("kernel,ape\n")
for kernel in kernels:
	tmp_cycles = cycles[df['appName'] == kernel]
	tmp_ape = np.mean(tmp_cycles['abe'])
	tmp_err_std = np.std(tmp_cycles['abe'])

	if kernel in pointer or kernel in extras:
	    continue
	print "%s:%f, %f" % (kernel, tmp_ape, tmp_err_std)
	f.write("%s,%f\n" % (kernel, tmp_ape))
f.close()


errors = []
for i in range(len(cycles['modelled_cycle'])):
        if cycles['appName'][i] in pointer or cycles['appName'][i] in extras:
            continue
 
        if cycles['coreF'][i] >= 500 and cycles['memF'][i] >= 500:
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
           

print "MAPE of %d samples: %f" % (len(errors), np.mean(errors))
