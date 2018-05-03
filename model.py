import pandas as pd
import numpy as np

gpucard = 'gtx980'

csv_perf = "%s-DVFS-Performance-cut.csv" % gpucard
df = pd.read_csv(csv_perf, header = 0)

print df.head(3)
print df.dtypes

a_L_DM = 222.78   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
b_L_DM = 277.32   # a * f_core / f_mem + b, a = 222.78, b = 277.32 for gtx980
L_L2 = 222   # 222 for gtx980
L_INST = 4   # 4 for gtx980
a_D_DM = 805.03    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
b_D_DM = 8.1762    # a / f_mem + b, a = 805.03, b = 8.1762 for gtx980
D_L2 = 1     # 1 for l2 cache
L_sh = 28    # 28 for gtx980
WARPS_MAX = 64 # 64 for gtx980
SM_COUNT = 16 # 56 for p100, 16 for gtx980
CORES_SM = 128 # 64 for p100, 128 for gtx980
WIDTH_MEM = 256 # 4096 for p100, 256 for gtx980

# df['n_gld'] = None
df['n_gld'] = None
df['n_gst'] = None
df['l2_hit'] = None
df['n_shm_ld'] = None
df['n_shm_st'] = None
df['insts'] = None
df['act_util'] = None
df['L_DM'] = None
df['D_DM'] = None

df['n_gld'] = df['l2_read_transactions'] / df['warps']
df['n_gst'] = df['l2_write_transactions'] / df['warps']

# df['l2_miss'] = df['dram_read_transactions'] / df['l2_read_transactions']
# df['l2_miss'] = df['dram_write_transactions'] / df['l2_write_transactions']
df['l2_miss'] = (df['dram_read_transactions'] + df['dram_write_transactions']) / (df['l2_read_transactions'] + df['l2_write_transactions'])
# df['l2_miss'][df['l2_miss'] >= 1] = 1
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

# cycles['offset'] = -cycles['cold_miss'] # dram latency is hidden
# cycles['offset'] = -cycles['sm_op']  # compute/shared cycle is hidden
# cycles['offset'] = -cycles['cold_miss'] - cycles['sm_op'] # compute/shared cycle and dram latency is hidden
# cycles['offset'] = -cycles['mem_op'] + cycles['lat_op'] # memory latency cannot be hidden
cycles['offset'] = 0

cycles['modelled_cycle'] = cycles['cold_miss'] + cycles['mem_op'] + cycles['sm_op'] + cycles['offset']

cycles['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (WARPS_MAX * SM_COUNT * df['act_util']))
cycles['abe'] = abs(cycles['modelled_cycle'] - cycles['real_cycle']) / cycles['real_cycle']

pointer = [	\
			# 'backprop', \
			# 'BlackScholes', \					# compute/shared cycle and dram latency is hidden
			# 'conjugateGradient', \			# dram latency is hidden
			# 'convolutionSeparable', \			# compute/shared cycle is hidden
			# 'convolutionTexture', \			# no hidden, but accuracy higher with l2 hit rate 0.3
			# 'fastWalshTransform', \			# compute/shared cycle is hidden
			# 'histogram', \					# maybe a mixture
			# 'hotspot', \						# memory latency cannot be hidden
			# 'matrixMul', \					# not a piece of cake
			# 'matrixMul(Global)', \			# compute/shared cycle and dram latency is hidden
			# 'MergeSort', \					# maybe a mixture
			# 'nn', \							# not a piece of cake
			# 'quasirandomGenerator', \			# compute/shared cycle and dram latency is hidden
			# 'reduction', \					# memory latency cannot be hidden
			# 'scalarProd', \					# compute/shared cycle and dram latency is hidden
			# 'scan', \							# compute/shared cycle and dram latency is hidden
			# 'SobolQRNG', \					# compute/shared cycle and dram latency is hidden
			# 'sortingNetworks', \				# compute/shared cycle and dram latency is hidden
			# 'transpose', \					# dram latency is hidden
			'vectorAdd'
			]
errors = []
for i in range(len(cycles['modelled_cycle'])):
	if df['appName'][i] in pointer: # and df['coreF'][i] == 500 and df['memF'][i] == 600:
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
		print 'modelled', cycles['modelled_cycle'][i]
		print 'real', cycles['real_cycle'][i], df['time/ms'][i], "ms"
		print 'relative error', cycles['abe'][i]
		print '\n'
		errors.append(cycles['abe'][i])

print "MAPE:", np.mean(errors)