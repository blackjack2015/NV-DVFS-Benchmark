import pandas as pd
import numpy as np
import sys
from settings import *

gpucard = 'gtx980'

csv_perf = "csvs/%s-DVFS-Performance-cut.csv" % gpucard
df = pd.read_csv(csv_perf, header = 0)

df['n_gld'] = df['l2_read_transactions'] / df['warps']
df['n_gst'] = df['l2_write_transactions'] / df['warps']

# df['l2_miss'] = df['dram_read_transactions'] / df['l2_read_transactions']
# df['l2_miss'] = df['dram_write_transactions'] / df['l2_write_transactions']
df['l2_miss'] = (df['dram_read_transactions'] + df['dram_write_transactions']) / (df['l2_read_transactions'] + df['l2_write_transactions'])
df.loc[df['l2_miss'] > 1, 'l2_miss'] = 1

df['l2_hit'] = 1 - df['l2_miss']
df['n_shm_ld'] = df['shared_load_transactions'] / df['warps']
df['n_shm_st'] = df['shared_store_transactions'] / df['warps']
df['mem_insts'] = df['n_gld'] + df['n_gst'] + df['n_shm_ld'] + df['n_shm_st']
df['insts'] = df['inst_per_warp'] - df['mem_insts']
df['act_util'] = df['achieved_occupancy']
df['L_DM'] = a_L_DM * df['coreF'] / df['memF'] + b_L_DM
df['D_DM'] = (a_D_DM / df['memF'] + b_D_DM) * df['coreF'] / df['memF']


cycles = pd.DataFrame(columns=['cold_miss', 'mem_op', 'sm_op', 'modelled_cycle', 'real_cycle']) # real_cycle per round
cycles['cold_miss'] = df['L_DM']
cycles['mem_op'] = (df['n_gld'] + df['n_gst']) * \
				   (df['D_DM'] * (1 - df['l2_hit']) + D_L2 * df['l2_hit']) \
				   * WARPS_MAX * df['act_util']
cycles['lat_op'] = (df['n_gld'] + df['n_gst']) * \
				   ((df['L_DM'] + df['D_DM']) * (1 - df['l2_hit']) + L_L2 * df['l2_hit'])
cycles['sm_op'] = (df['n_shm_ld'] + df['n_shm_st']) * L_sh \
					+ df['insts'] * L_INST

# add type for offset
cycles['offset'] = None
for idx, item in df.iterrows():
	cur_name = df['appName'][idx]
	if eqType[cur_name] == DM_HID:
		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'cold_miss']
	elif eqType[cur_name] == COMP_HID:
		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'sm_op']
	elif eqType[cur_name] == DM_COMP_HID:
		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'cold_miss'] -cycles.loc[idx, 'sm_op']
	elif eqType[cur_name] == MEM_LAT_BOUND:
		cycles.loc[idx, 'offset'] = -cycles.loc[idx, 'mem_op'] +cycles.loc[idx, 'lat_op']
	elif eqType[cur_name] == NO_HID:
		cycles.loc[idx, 'offset'] = 0
	elif eqType[cur_name] == MIX:
		cycles.loc[idx, 'offset'] = 0
	else:
		print "Invalid modeling type..."
		sys.exit(-1)	

cycles['modelled_cycle'] = cycles['cold_miss'] + cycles['mem_op'] + cycles['sm_op'] + cycles['offset']
cycles['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (WARPS_MAX * SM_COUNT * df['act_util']))
cycles['abe'] = abs(cycles['modelled_cycle'] - cycles['real_cycle']) / cycles['real_cycle']

pointer = ['backprop', 'matrixMul', 'nn']

errors = []
for i in range(len(cycles['modelled_cycle'])):
	# if df['appName'][i] in pointer and df['coreF'][i] >= 500 and df['memF'][i] >= 500:
	if df['coreF'][i] >= 500 and df['memF'][i] >= 500:
		print i
		print 'n_gld', df['n_gld'][i]
		print 'n_gst', df['n_gst'][i]
		print 'l2_hit', df['l2_hit'][i]
		print 'n_shm_ld', df['n_shm_ld'][i]
		print 'n_shm_st', df['n_shm_st'][i]
		print 'insts', df['insts'][i]
		print 'act_util', df['act_util'][i]
		print 'dram delay', df['D_DM'][i]
		print 'dram latency', df['L_DM'][i]
		print 'coreF', df['coreF'][i]
		print 'memF', df['memF'][i]
		print 'mem_op', cycles['mem_op'][i]
		print 'sm_op', cycles['sm_op'][i]
		print 'lat_op', cycles['lat_op'][i]
		print 'modelled', cycles['modelled_cycle'][i]
		print 'real', cycles['real_cycle'][i], df['time/ms'][i], "ms"
		print 'relative error', cycles['abe'][i]
		print '\n'
		errors.append(cycles['abe'][i])

print "MAPE of %d samples: %f" % (len(errors), np.mean(errors))
