import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data-root', type=str, help='data file path', default='raw')
parser.add_argument('--benchmark-setting', type=str, help='gpu and dvfs setting', default='gtx980-low-dvfs')
parser.add_argument('--kernel-setting', type=str, help='kernel list', default='real-small-workload')

opt = parser.parse_args()
print opt

perf_data = pd.read_csv("csvs/%s/%s-%s-Performance.csv" % (opt.data_root, opt.benchmark_setting, opt.kernel_setting))
power_data = pd.read_csv("csvs/%s/%s-%s-Power.csv" % (opt.data_root, opt.benchmark_setting, opt.kernel_setting))

perf_data = perf_data.sort_values(by=['appName', 'coreF', 'memF'])
power_data = power_data.sort_values(by=['appName', 'coreF', 'memF'])

perf_data['power/W'] = power_data['power/W']
perf_data.to_csv("csvs/%s/%s-%s-Performance-Power.csv" % (opt.data_root, opt.benchmark_setting, opt.kernel_setting))
