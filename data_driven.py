import pandas as pd
import numpy as np
import sys
from settings import *

df = pd.read_csv(csv_perf, header = 0)

params = pd.DataFrame(columns=['n_shm_ld', 'n_shm_st', 'n_gld', 'n_gst', 'n_dm_ld', 'n_dm_st', 'n_flop_sp', 'mem_insts', 'insts']) 

# shared memory information
params['n_shm_ld'] = df['shared_load_transactions'] / df['warps']
params['n_shm_st'] = df['shared_store_transactions'] / df['warps']

# global memory information
params['n_gld'] = df['l2_read_transactions'] / df['warps']
params['n_gst'] = df['l2_write_transactions'] / df['warps']

# dram memory information
params['n_dm_ld'] = df['dram_read_transactions'] / df['warps']
params['n_dm_st'] = df['dram_write_transactions'] / df['warps']

# compute insts
params['n_flop_sp'] = df['flop_count_sp'] / df['warps']

# other parameters
params['mem_insts'] = params['n_gld'] + params['n_gst'] + params['n_shm_ld'] + params['n_shm_st']
params['insts'] = df['inst_per_warp'] - params['mem_insts']

# grouth truth cycle per SM per round
params['real_cycle'] = df['time/ms'] * df['coreF'] * 1000 / (df['warps'] / (WARPS_MAX * SM_COUNT * df['achieved_occupancy']))

# normalize
params = params.div(params.loc[:, params.columns != 'real_cycle'].sum(axis=1), axis=0)

# frequency ratio, core/mem
params['c_to_m'] = df['coreF'] * 1.0 / df['memF']

print params.head(10)


