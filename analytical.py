import pandas as pd
import numpy as np
import sys
from settings import *

gpucard = 'p100'
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
#features['n_gld'] = df['l2_read_transactions'] / df['warps']
#features['n_gst'] = df['l2_write_transactions'] / df['warps']
features['n_gld'] = df['gld_transactions'] / df['warps']
features['n_gst'] = df['gst_transactions'] / df['warps']
#features['n_gld'] = (df['l2_read_transactions'] + df['shared_load_transactions']) / df['warps']
#features['n_gst'] = (df['l2_write_transactions'] + df['shared_store_transactions']) / df['warps']

# l2 information
#features['l2_miss'] = df['dram_read_transactions'] / df['l2_read_transactions']
#features['l2_miss'] = df['dram_write_transactions'] / df['l2_write_transactions']
features['l2_miss'] = (df['dram_read_transactions'] + df['dram_write_transactions']) / (df['l2_read_transactions'] + df['l2_write_transactions'])
# df['l2_miss'] = (df['dram_read_transactions'] + df['dram_write_transactions']) / ((df['n_gst'] + df['n_gld']) * df['warps'])
features.loc[features['l2_miss'] > 1, 'l2_miss'] = 1
features['l2_hit'] = 1 - features['l2_miss']

# other parameters
features['mem_insts'] = features['n_gld'] + features['n_gst'] + features['n_shm_ld'] + features['n_shm_st']
features['insts'] = df['inst_per_warp'] - features['mem_insts']
features['act_util'] = df['achieved_occupancy']
features['L_DM'] = GPUCONF.a_L_DM * df['coreF'] / df['memF'] + GPUCONF.b_L_DM
features['D_DM'] = (GPUCONF.a_D_DM / df['memF'] + GPUCONF.b_D_DM) * df['coreF'] / df['memF']

# other methodology
def hong2009(df):
    pass

def song2013(df):
    pass

def qiang2018(df):

    # analytical model
    cycles = pd.DataFrame(columns=['appName', 'coreF', 'memF', 'cold_miss', 'mem_op', 'sm_op', 'modelled_cycle', 'real_cycle']) # real_cycle per round
    cycles['appName'] = df['appName']
    cycles['coreF'] = df['coreF']
    cycles['memF'] = df['memF']
    cycles['cold_miss'] = df['L_DM']
    cycles['mem_op'] = (df['n_gld'] + df['n_gst']) * (df['D_DM'] * (1 - df['l2_hit']) + GPUCONF.D_L2 * df['l2_hit']) * GPUCONF.WARPS_MAX * df['act_util']
    cycles['lat_op'] = (df['n_gld'] + df['n_gst']) * ((df['L_DM'] + df['D_DM']) * (1 - df['l2_hit']) + GPUCONF.L_L2 * df['l2_hit'])
    cycles['shm_op'] = (df['n_shm_ld'] + df['n_shm_st']) * GPUCONF.L_sh
    cycles['compute_lat'] = df['insts'] * GPUCONF.L_INST
    cycles['sm_op'] = cycles['shm_op'] + cycles['compute_lat']
    cycles['compute'] = df['insts'] * df['act_util'] * GPUCONF.WARPS_MAX + GPUCONF.L_INST
    #cycles['sm_op'] = df['insts'] * L_INST
    
    # add type for offset
    cycles['offset'] = None
    for idx, item in df.iterrows():
    	cur_name = df['appName'][idx]
    	if GPUCONF.eqType[cur_name] == DM_HID:
    		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'cold_miss']
    	elif GPUCONF.eqType[cur_name] == COMP_HID:
    		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'sm_op']
    	elif GPUCONF.eqType[cur_name] == MEM_HID:
    		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'mem_op']
    	elif GPUCONF.eqType[cur_name] == DM_COMP_HID:
    		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'cold_miss'] -cycles.loc[idx, 'sm_op']
    	elif GPUCONF.eqType[cur_name] == MEM_LAT_BOUND:
    		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'mem_op'] -cycles.loc[idx, 'cold_miss'] +cycles.loc[idx, 'lat_op']
    	elif GPUCONF.eqType[cur_name] == NO_HID:
    		cycles.loc[idx, 'offset'] = 0
    	elif GPUCONF.eqType[cur_name] == MIX:
    		cycles.loc[idx, 'offset'] = 0
    	else:
    		print "Invalid modeling type of %s..." % cur_name
    		sys.exit(-1)	
    
    cycles['modelled_cycle'] = cycles['cold_miss'] + cycles['mem_op'] + cycles['sm_op'] + cycles['offset']
    return cycles


cycles = qiang2018(features)
cycles['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (GPUCONF.WARPS_MAX * GPUCONF.SM_COUNT * df['achieved_occupancy']))
cycles['abe'] = abs(cycles['modelled_cycle'] - cycles['real_cycle']) / cycles['real_cycle']

# save results to csv/xlsx
cycles.to_csv("%s-cycles.csv" % gpucard)
writer = pd.ExcelWriter("%s-cycles.xlsx" % gpucard)
cycles.to_excel(writer, 'Sheet1')
writer.save()

# pointer = ['backprop', 'matrixMul', 'nn']
pointer = ['histogram']
#kernels = df['appName']
#pointer = [kernels[16]]
# pointer = ['transpose']
kernels = features['appName'].drop_duplicates()

#for kernel in kernels:
#	tmp_cycles = cycles[df['appName'] == kernel]
#	tmp_ape = np.mean(tmp_cycles['abe'])
#
#	if kernel in pointer:
#		print "big bias: %s:%f." % (kernel, tmp_ape)
#		continue
#	print "%s:%f." % (kernel, tmp_ape)


errors = []
for i in range(len(cycles['modelled_cycle'])):
	# if df['appName'][i] not in pointer and df['coreF'][i] >= 500 and df['memF'][i] >= 500:
	# if df['appName'][i] not in pointer:
        #if cycles['appName'][i] in pointer and cycles['coreF'][i] >= 500 and cycles['memF'][i] >= 500:
	if cycles['coreF'][i] >= 500 and cycles['memF'][i] >= 500:
		if True or cycles['abe'][i] > 0.20:
		    print cycles['appName'][i], cycles['coreF'][i], cycles['memF'][i]
		    #print i, df['appName'][i], 'relative error', cycles['abe'][i]
		    # print i, df['appName'][i]
		    print 'n_gld', features['n_gld'][i]
		    print 'n_gst', features['n_gst'][i]
		    print 'l2_hit', features['l2_hit'][i]
		    print 'n_shm_ld', features['n_shm_ld'][i]
		    print 'n_shm_st', features['n_shm_st'][i]
		    print 'insts', features['insts'][i]
		    print 'act_util', features['act_util'][i]
		    print 'dram delay', features['D_DM'][i]
		    print 'dram latency', features['L_DM'][i]
		    print 'coreF', features['coreF'][i]
		    print 'memF', features['memF'][i]
		    print 'mem_op', cycles['mem_op'][i]
		    print 'sm_op', cycles['sm_op'][i]
		    print 'lat_op', cycles['lat_op'][i]
		    print 'modelled', cycles['modelled_cycle'][i]
		    print 'real', cycles['real_cycle'][i], df['time/ms'][i], "ms"
		    print 'relative error', cycles['abe'][i]
		    print '\n'
		errors.append(cycles['abe'][i])
           

print "MAPE of %d samples: %f" % (len(errors), np.mean(errors))
