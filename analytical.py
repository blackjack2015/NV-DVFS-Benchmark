import pandas as pd
import numpy as np
import sys
from settings import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpucard', type=str, help='gpu card', default='gtx980')

opt = parser.parse_args()
print opt

gpucard = opt.gpucard
csv_perf = "csvs/v0/%s-DVFS-Performance.csv" % gpucard
df = pd.read_csv(csv_perf, header = 0)

if gpucard == 'gtx980':
    GPUCONF = GTX980()
elif gpucard == 'titanx':
    GPUCONF = TITANX()
elif gpucard == 'p100':
    GPUCONF = P100()

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

# other parameters
features['mem_insts'] = features['n_gld'] + features['n_gst'] + features['n_shm_ld'] + features['n_shm_st'] / 4.0
features['insts'] = df['inst_per_warp'] - features['mem_insts']
features.loc[features['insts'] < 0, 'insts'] = 0
features['act_util'] = df['achieved_occupancy']
features['L_DM'] = GPUCONF.a_L_DM * df['coreF'] / df['memF'] + GPUCONF.b_L_DM
features['D_DM'] = (GPUCONF.a_D_DM / df['memF'] + GPUCONF.b_D_DM) * df['coreF'] / df['memF']

# save featuress to csv/xlsx
features.to_csv("%s-features.csv" % gpucard)
writer = pd.ExcelWriter("%s-features.xlsx" % gpucard)
features.to_excel(writer, 'Sheet1')
writer.save()

# other methodology
def hong2009(df):
    pass

def song2013(df):
    pass

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
    cycles['shm_del'] = (df['n_shm_ld'] + df['n_shm_st']) * df['act_util'] * GPUCONF.WARPS_MAX + GPUCONF.L_sh # shared queue delay for all warps per round
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


cycles = qiang2018(features)
cycles['exec_rounds'] = df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy'])
#cycles['exec_rounds'] = cycles['exec_rounds'].astype(int)
cycles['real_cycle'] = df['time/ms'] * df['coreF'] * 1000.0 / cycles['exec_rounds']
cycles['abe'] = abs(cycles['modelled_cycle'] - cycles['real_cycle']) / cycles['real_cycle']

# save results to csv/xlsx
cycles.to_csv("%s-cycles.csv" % gpucard)
writer = pd.ExcelWriter("%s-cycles.xlsx" % gpucard)
cycles.to_excel(writer, 'Sheet1')
writer.save()

# pointer = ['backprop', 'matrixMul', 'nn']
pointer = ['convolutionTexture', 'nn', 'SobolQRNG', 'reduction', 'hotspot'] #, 'backprop', 'conjugateGradient', 'mergeSort', 'quasirandomGenerator', 'scalarProd']
#kernels = df['appName']
#pointer = [kernels[16]]
# pointer = ['transpose']
kernels = features['appName'].drop_duplicates()
kernels.sort()

for kernel in kernels:
	tmp_cycles = cycles[df['appName'] == kernel]
	tmp_ape = np.mean(tmp_cycles['abe'])
	tmp_err_std = np.std(tmp_cycles['abe'])

	if kernel in pointer:
	    continue
	print "%s:%f, %f" % (kernel, tmp_ape, tmp_err_std)


errors = []
for i in range(len(cycles['modelled_cycle'])):
	# if df['appName'][i] not in pointer and df['coreF'][i] >= 500 and df['memF'][i] >= 500:
	# if df['appName'][i] not in pointer:
        if cycles['appName'][i] not in pointer and cycles['coreF'][i] >= 500 and cycles['memF'][i] >= 500:
	#if cycles['coreF'][i] >= 500 and cycles['memF'][i] >= 500:
		#if True or cycles['abe'][i] > 0.20:
		#    print cycles['appName'][i], cycles['coreF'][i], cycles['memF'][i]
		#    #print i, df['appName'][i], 'relative error', cycles['abe'][i]
		#    # print i, df['appName'][i]
		#    print 'n_gld', features['n_gld'][i]
		#    print 'n_gst', features['n_gst'][i]
		#    print 'l2_hit', features['l2_hit'][i]
		#    print 'n_shm_ld', features['n_shm_ld'][i]
		#    print 'n_shm_st', features['n_shm_st'][i]
		#    print 'insts', features['insts'][i]
		#    print 'act_util', features['act_util'][i]
		#    print 'dram delay', features['D_DM'][i]
		#    print 'dram latency', features['L_DM'][i]
		#    print 'coreF', features['coreF'][i]
		#    print 'memF', features['memF'][i]
		#    print 'mem_del', cycles['mem_del'][i]
		#    print 'sm_del', cycles['sm_del'][i]
		#    print 'modelled', cycles['modelled_cycle'][i]
		#    print 'real', cycles['real_cycle'][i], df['time/ms'][i], "ms"
		#    print 'relative error', cycles['abe'][i]
		#    print '\n'
		errors.append(cycles['abe'][i])
           

print "MAPE of %d samples: %f" % (len(errors), np.mean(errors))
