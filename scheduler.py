import os,sys
import argparse
import subprocess
import time
import re
import configparser
import json

parser = argparse.ArgumentParser()
parser.add_argument('--benchmark-setting', type=str, help='gpu benchmark setting', default='p100')
parser.add_argument('--kernel-setting', type=str, help='kernels of benchmark', default='synthetic')
parser.add_argument('--trace', type=str, help='trace of tasks', default='trace1')
parser.add_argument('--app-root', type=str, help='folder of applications', default='applications/linux')

opt = parser.parse_args()
print(opt)

BS_SETTING = '%s.cfg' % opt.benchmark_setting
KS_SETTING = '%s.cfg' % opt.kernel_setting

APP_ROOT = opt.app_root
LOG_ROOT = 'schedule/%s-%s' % (opt.benchmark_setting, opt.kernel_setting)

try:
    os.makedirs(LOG_ROOT)
except OSError:
    pass

# Reading benchmark settings
cf_bs = configparser.ConfigParser()
cf_bs.read("configs/benchmarks/%s" % BS_SETTING)
cf_ks = configparser.ConfigParser()
cf_ks.read("configs/kernels/%s" % KS_SETTING)

nvIns_dev_id = cf_bs.getint("profile_control", "nvIns_device_id")
cuda_dev_id = cf_bs.getint("profile_control", "cuda_device_id")
pw_sample_int = cf_bs.getint("profile_control", "power_sample_interval")
core_frequencies = json.loads(cf_bs.get("dvfs_control", "coreF"))
memory_frequencies = json.loads(cf_bs.get("dvfs_control", "memF"))
powerState = cf_bs.getint("dvfs_control", "powerState")
if powerState == 5:
    freqState = 1
else:
    freqState = powerState

# Read GPU application settings
benchmark_programs = cf_ks.sections()

pw_sampling_cmd = 'nohup ./nvml_samples.exe -device=%d -si=%d -output=%s/%s 1>null 2>&1 &'
app_exec_cmd = './%s/%s.exe %s -device=%d -secs=%d >> %s/%s'
kill_pw_cmd = 'killall nvml_samples.exe'
if powerState !=0:
    dvfs_cmd = './nvidiaInspector.exe -forcepstate:%s,%d -setGpuClock:%s,%d,%s -setMemoryClock:%s,%d,%s' \
                    % (nvIns_dev_id, powerState, nvIns_dev_id, freqState, '%s', nvIns_dev_id, freqState, '%s')
else:
    dvfs_cmd = './nvidiaInspector.exe -setBaseClockOffset:%s,%d,%s -setMemoryClockOffset:%s,%d,%s' \
                    % (nvIns_dev_id, freqState, '%s', nvIns_dev_id, freqState, '%s')

# start record power data
command = pw_sampling_cmd % (nvIns_dev_id, pw_sample_int, LOG_ROOT, 'power_%s.log'%opt.trace)
print(command)
os.system(command)
with open(opt.trace, "r") as f:
    tasks = f.readlines()
tasks = [item.strip() for item in tasks]
start = time.time()
for i, task in enumerate(tasks):

    print(task)
    app, core_f, mem_f, running_time = task.split(" ")
    core_f = int(core_f)
    mem_f = int(mem_f)
    running_time = int(running_time)
    # set specific frequency
    command = dvfs_cmd % (core_f, mem_f)
    
    print(command)
    os.system(command)

    arg = json.loads(cf_ks.get(app, 'args'))[0]

    # arg, number = re.subn('-device=[0-9]*', '-device=%d' % cuda_dev_id, arg)
    perflog = 'benchmark_%s_core%d_mem%d_perf.log' % (app, core_f, mem_f)

    # execute program to collect power data
    os.system("echo \"arg:%s\" > %s/%s" % (arg, LOG_ROOT, perflog))
    command = app_exec_cmd % (APP_ROOT, app, arg, cuda_dev_id, running_time, LOG_ROOT, perflog)
    print(command)
    os.system(command)

# stop record power data
def get_aver_power():
    with open(os.path.join(LOG_ROOT, 'power_%s.log' % opt.trace), 'r') as f:
        lines = f.readlines()[3:]
    powers = [int(line.split()[-1].strip()) for line in lines]
    return sum(powers) / 1000.0 / len(powers)
os.system(kill_pw_cmd)
total_time = time.time() - start
aver_power = get_aver_power()
energy = total_time * aver_power
print("Total running time is: %.2f s." % total_time)
print("Average power is: %.2f W." % aver_power)
print("Total energy is %.2f J." % energy)
