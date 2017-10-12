import os
import subprocess
import time
import re
import ConfigParser
import json

APP_ROOT = 'applications'
LOG_ROOT = 'logs'

# Reading benchmark settings
cf_bs = ConfigParser.SafeConfigParser()
cf_bs.read("configs/benchmark_settings.cfg")

running_iters = cf_bs.getint("profile_control", "iters")
running_time = cf_bs.getint("profile_control", "secs")
nvIns_dev_id = cf_bs.getint("profile_control", "nvIns_device_id")
cuda_dev_id = cf_bs.getint("profile_control", "cuda_device_id")
pw_sample_int = cf_bs.getint("profile_control", "power_sample_interval")
rest_int = cf_bs.getint("profile_control", "rest_time")
metrics = json.loads(cf_bs.get("profile_control", "metrics"))
core_frequencies = json.loads(cf_bs.get("dvfs_control", "coreF"))
memory_frequencies = json.loads(cf_bs.get("dvfs_control", "memF"))

# Read GPU application settings
cf_ks = ConfigParser.SafeConfigParser()
cf_ks.read("configs/kernels_settings.cfg")
benchmark_programs = cf_ks.sections()

print benchmark_programs
print metrics
print core_frequencies
print memory_frequencies

# time.sleep(100)
# open GPU monitor
# os.system('start nvidiaInspector.exe -showMonitoring')

# reset GPU first
command = 'nvidiaInspector.exe -forcepstate:%s,16' % nvIns_dev_id
print command
os.system(command)
time.sleep(rest_int)

for core_f in core_frequencies:
    for mem_f in memory_frequencies:

        # set specific frequency
        command = 'nvidiaInspector.exe -forcepstate:%s,5 -setMemoryClock:%s,1,%s -setGpuClock:%s,1,%s' \
                        % (nvIns_dev_id, nvIns_dev_id, mem_f, nvIns_dev_id, core_f)
        
        print command
        os.system(command)
        time.sleep(rest_int)

        for app in benchmark_programs:

            args = json.loads(cf_ks.get(app, 'args'))

            argNo = 0

            for arg in args:

                # arg, number = re.subn('-device=[0-9]*', '-device=%d' % cuda_dev_id, arg)
                powerlog = 'benchmark_%s_core%d_mem%d_input%02d_power.log' % (app, core_f, mem_f, argNo)
                perflog = 'benchmark_%s_core%d_mem%d_input%02d_perf.log' % (app, core_f, mem_f, argNo)
                metricslog = 'benchmark_%s_core%d_mem%d_input%02d_metrics.log' % (app, core_f, mem_f, argNo)


                # # start record power data
                # os.system("echo \"arg:%s\" >> %s/%s" % (arg, LOG_ROOT, powerlog))
                # command = 'start /B nvml_samples.exe -device=%d -output=%s/%s > nul' % (nvIns_dev_id, LOG_ROOT, powerlog)
                # print command
                # os.system(command)
                # time.sleep(rest_int)

                # # execute program to collect power data
                # os.system("echo \"arg:%s\" >> %s/%s" % (arg, LOG_ROOT, perflog))
                # command = '%s\\%s %s -device=%d -secs=%d >> %s/%s' % (APP_ROOT, app, arg, cuda_dev_id, running_time, LOG_ROOT, perflog)
                # print command
                # os.system(command)
                # time.sleep(rest_int)

                # # stop record power data
                # os.system('tasklist|findstr "nvml_samples.exe" && taskkill /F /IM nvml_samples.exe')


                # execute program to collect time data
                # arg, number = re.subn('-iters=[0-9]*', '-iters=10', arg)
                command = 'nvprof --profile-child-processes %s/%s %s -device=%d -iters=5 >> %s/%s 2>&1' % (APP_ROOT, app, arg, cuda_dev_id, LOG_ROOT, perflog)
                print command
                os.system(command)
                time.sleep(rest_int)

                # collect grid and block settings
                command = 'nvprof --print-gpu-trace --profile-child-processes %s/%s %s -device=%d -iters=15 >> %s/%s 2>&1' % (APP_ROOT, app, arg, cuda_dev_id, LOG_ROOT, metricslog)
                print command
                os.system(command)
                time.sleep(rest_int)

                # execute program to collect metrics data
                metCount = 0

                # to be fixed, the stride should be a multiplier of the metric number
                while metCount < len(metrics):

                    if metCount + 3 > len(metrics):
                        metStr = ','.join(metrics[metCount:])
                    else:
                        metStr = ','.join(metrics[metCount:metCount + 3])
                    command = 'nvprof --devices %s --metrics %s %s/%s %s -device=%d -iters=15 >> %s/%s 2>&1' % (cuda_dev_id, metStr, APP_ROOT, app, arg, cuda_dev_id, LOG_ROOT, metricslog)
                    print command
                    os.system(command)
                    time.sleep(rest_int)
                    metCount += 3

# reset GPU first
command = 'nvidiaInspector.exe -forcepstate:%s,16' % nvIns_dev_id
print command
os.system(command)
time.sleep(rest_int)
os.system('pause')