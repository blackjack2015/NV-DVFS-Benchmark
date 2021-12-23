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
parser.add_argument('--mode', type=str, help='benchmark mode', default='fast', choices=['fast', 'verbose'])
parser.add_argument('--app-root', type=str, help='folder of applications', default='applications/linux')

opt = parser.parse_args()
print(opt)

BS_SETTING = '%s.cfg' % opt.benchmark_setting
KS_SETTING = '%s.cfg' % opt.kernel_setting

APP_ROOT = opt.app_root
LOG_ROOT = 'logs/%s-%s' % (opt.benchmark_setting, opt.kernel_setting)

try:
    os.makedirs(LOG_ROOT)
except OSError:
    pass

# Reading benchmark settings
cf_bs = configparser.ConfigParser()
cf_bs.read("configs/benchmarks/%s" % BS_SETTING)
cf_ks = configparser.ConfigParser()
cf_ks.read("configs/kernels/%s" % KS_SETTING)

running_iters = cf_bs.getint("profile_control", "iters")
running_time = cf_bs.getint("profile_control", "secs")
nvIns_dev_id = cf_bs.getint("profile_control", "nvIns_device_id")
cuda_dev_id = cf_bs.getint("profile_control", "cuda_device_id")
pw_sample_int = cf_bs.getint("profile_control", "power_sample_interval")
rest_int = cf_bs.getint("profile_control", "rest_time")
metrics = json.loads(cf_bs.get("profile_control", "metrics"))
core_frequencies = json.loads(cf_bs.get("dvfs_control", "coreF"))
memory_frequencies = json.loads(cf_bs.get("dvfs_control", "memF"))
powerState = cf_bs.getint("dvfs_control", "powerState")
if powerState == 5:
    freqState = 1
else:
    freqState = powerState

# Read GPU application settings
benchmark_programs = cf_ks.sections()

print(benchmark_programs)
print(metrics)
print(core_frequencies)
print(memory_frequencies)

pw_sampling_cmd = 'nohup ./nvml_samples.exe -device=%d -si=%d -output=%s/%s 1>null 2>&1 &'
app_exec_cmd = './%s/%s.exe %s -device=%d -secs=%d >> %s/%s'
kill_pw_cmd = 'killall nvml_samples.exe'
if powerState !=0:
    dvfs_cmd = './nvidiaInspector.exe -forcepstate:%s,%d -setGpuClock:%s,%d,%s -setMemoryClock:%s,%d,%s' \
                    % (nvIns_dev_id, powerState, nvIns_dev_id, freqState, '%s', nvIns_dev_id, freqState, '%s')
else:
    dvfs_cmd = './nvidiaInspector.exe -setBaseClockOffset:%s,%d,%s -setMemoryClockOffset:%s,%d,%s' \
                    % (nvIns_dev_id, freqState, '%s', nvIns_dev_id, freqState, '%s')

for core_f in core_frequencies:
    for mem_f in memory_frequencies:

        # set specific frequency
        command = dvfs_cmd % (core_f, mem_f)
        
        print(command)
        os.system(command)
        time.sleep(rest_int)

        for i, app in enumerate(benchmark_programs):

            #if i <= 18:
            #    continue
            args = json.loads(cf_ks.get(app, 'args'))

            for argNo, arg in enumerate(args):

                # arg, number = re.subn('-device=[0-9]*', '-device=%d' % cuda_dev_id, arg)
                powerlog = 'benchmark_%s_core%d_mem%d_input%02d_power.log' % (app, core_f, mem_f, argNo)
                perflog = 'benchmark_%s_core%d_mem%d_input%02d_perf.log' % (app, core_f, mem_f, argNo)
                metricslog = 'benchmark_%s_core%d_mem%d_input%02d_metrics.log' % (app, core_f, mem_f, argNo)

                # start record power data
                os.system("echo \"arg:%s\" > %s/%s" % (arg, LOG_ROOT, powerlog))
                command = pw_sampling_cmd % (nvIns_dev_id, pw_sample_int, LOG_ROOT, powerlog)
                print(command)
                os.system(command)
                time.sleep(rest_int)

                # execute program to collect power data
                os.system("echo \"arg:%s\" > %s/%s" % (arg, LOG_ROOT, perflog))
                command = app_exec_cmd % (APP_ROOT, app, arg, cuda_dev_id, running_time, LOG_ROOT, perflog)
                print(command)
                os.system(command)
                time.sleep(rest_int)

                # stop record power data
                os.system(kill_pw_cmd)

                # execute program to collect time data
                command = 'nvprof.exe --profile-child-processes %s/%s %s -device=%d -secs=5 >> %s/%s 2>&1' % (APP_ROOT, app, arg, cuda_dev_id, LOG_ROOT, perflog)
                print(command)
                os.system(command)
                time.sleep(rest_int)

                if opt.mode == 'verbose':
                    # collect grid and block settings
                    command = 'nvprof.exe --print-gpu-trace --profile-child-processes %s/%s %s -device=%d -iters=10 > %s/%s 2>&1' % (APP_ROOT, app, arg, cuda_dev_id, LOG_ROOT, metricslog)
                    print(command)
                    os.system(command)
                    time.sleep(rest_int)

                    # execute program to collect metrics data
                    metCount = 0

                    # to be fixed, the stride should be a multiplier of the metric number
                    stride = 2
                    while metCount < len(metrics):

                        if metCount + stride > len(metrics):
                            metStr = ','.join(metrics[metCount:])
                        else:
                            metStr = ','.join(metrics[metCount:metCount + stride])
                        command = 'nvprof.exe --devices %s --metrics %s %s/%s %s -device=%d -iters=10 >> %s/%s 2>&1' % (cuda_dev_id, metStr, APP_ROOT, app, arg, cuda_dev_id, LOG_ROOT, metricslog)
                        print(command)
                        os.system(command)
                        time.sleep(1)
                        metCount += stride


time.sleep(rest_int)
