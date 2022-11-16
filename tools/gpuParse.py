import time, argparse
import datetime
import sys
import csv
import os, glob, re
import numpy as np
import json
import pandas as pd
from log_parsers import parse_perf_log, parse_power_log, parse_metrics_log, parse_dcgm_log

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark-setting', type=str, help='gpu benchmark setting', default='v100-dvfs')
parser.add_argument('--kernel-setting', type=str, help='kernels of benchmark', default='real')
parser.add_argument('--core-base', type=int, help='base core frequency', default=0)
parser.add_argument('--mem-base', type=int, help='base memory frequency', default=0)

opt = parser.parse_args()
print(opt)

gpucard = opt.benchmark_setting
version = opt.kernel_setting
coreBase = opt.core_base
memBase = opt.mem_base

logRoot = 'logs/%s-%s' % (gpucard, version)
output_name = 'data/%s-%s-profile.csv' % (gpucard, version)

perf_filelist = glob.glob(r'%s/*perf.log' % logRoot)
perf_filelist.sort()


rows = pd.DataFrame()
for perf_file in perf_filelist:
    
    power_file = perf_file.replace('perf', 'power')
    dcgm_file = perf_file.replace('perf', 'dcgm')

    perf_dict = parse_perf_log(perf_file)
    pow_dict = parse_power_log(power_file)
    dcgm_dict = parse_dcgm_log(dcgm_file)

    one_data = {}
    one_data.update(perf_dict)
    one_data.update(pow_dict)
    one_data.update(dcgm_dict)

    print(one_data)

    rows = rows.append(one_data, ignore_index=True)

print(rows.head())
rows.to_csv(output_name, index=False)

