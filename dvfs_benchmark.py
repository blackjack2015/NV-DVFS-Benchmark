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

time.sleep(100)
# open GPU monitor
os.system('start nvidiaInspector.exe -showMonitoring')

# reset GPU first
command = 'nvidiaInspector.exe -forcepstate:%s,16' % nvIns_dev_id
print command
os.system(command)
time.sleep(wt)

for core_f in core_frequencies:
    for mem_f in memory_frequencies:

        # set specific frequency
        command = 'nvidiaInspector.exe -forcepstate:%s,5 -setMemoryClock:%s,1,%s -setGpuClock:%s,1,%s' % (nvIns_dev_id, \
                                                                                                        nvIns_dev_id, mem_f, \
                                                                                                        nvIns_dev_id, core_f)
        print command
        os.system(command)
        time.sleep(wt)

        for app in benchmark_programs:

            program_args = [item.replace('\n', '') for item in open(conf_args+app+'_args.txt').readlines()]

            for arg in program_args:

                # start record power data
                os.system('echo sm_clock:%s,mem_clock:%s >> dvfs.txt' % (core_f, mem_f))
                os.system('echo program:%s,arguments:%s >> dvfs.txt' % (app, arg))
                command = 'start /B nvidia-smi.exe -l 2 >> dvfs.txt'
                print command
                os.system(command)
                time.sleep(wt)

                os.system('echo sm_clock:%s,mem_clock:%s >> %s%s.txt' % (core_f, mem_f, log_folder.replace('/', '\\'), app))
                os.system('echo program:%s,arguments:%s >> %s%s.txt' % (app, arg, log_folder.replace('/', '\\'), app))
                # execute program to collect power data
                command = '%s%s %s >> %s%s.txt' % (app_folder.replace('/', '\\'), app, arg, log_folder.replace('/', '\\'), app)
                print command
                os.system(command)
                time.sleep(wt)

                # stop record power data
                os.system('tasklist|findstr "nvidia-smi.exe" && taskkill /F /IM nvidia-smi.exe')

                # execute program to collect time data
                arg, number = re.subn('-iters=[0-9]*', '-iters=50', arg)
                command = 'nvprof %s%s %s >> %s%s.txt 2>&1' % (app_folder.replace('/', '\\'), app, arg, log_folder.replace('/', '\\'), app)
                print command
                os.system(command)
                time.sleep(wt)

                # execute program to collect metrics data
                command = 'nvprof --devices %s --metrics %s %s%s %s >> %s%s.txt 2>&1' % (cuda_dev_id, ''.join(metrics), app_folder.replace('/', '\\'), app, arg, log_folder.replace('/', '\\'), app)
                print command
                os.system(command)
                time.sleep(wt)

# reset GPU first
command = 'nvidiaInspector.exe -forcepstate:%s,16' % nvIns_dev_id
print command
os.system(command)
time.sleep(wt)
os.system('pause')