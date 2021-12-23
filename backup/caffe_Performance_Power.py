import os
import subprocess
import time
import re

app_folder = 'Applications/'
log_folder = 'Logs/performance_power/'
pw_log_folder = 'Logs/power/'
conf_folder = 'Conf/'
conf_args = 'Conf/arguments/'
nvIns_dev_id = '0'
cuda_dev_id = '1'
wt = 15
st = 2

benchmark_programs = [item.replace('\n', '') for item in open(conf_folder+'applications.txt').readlines()]
core_frequencies = [item.replace('\n', '') for item in open(conf_folder+'core_frequency.txt').readlines()]
memory_frequencies = [item.replace('\n', '') for item in open(conf_folder+'memory_frequency.txt').readlines()]
metrics = [item.replace('\n', '') for item in open(conf_folder+'metrics.txt').readlines()]

print benchmark_programs
print core_frequencies
print memory_frequencies

# open GPU monitor
os.system('start nvidiaInspector.exe -showMonitoring')

# reset GPU first
command = 'nvidiaInspector.exe -forcepstate:%s,16' % nvIns_dev_id
print command
os.system(command)
time.sleep(wt)

arg_template = '-hA=%s -hB=%s -w=%s -device=1 -transpose=%s'
hA_set = [2**(10+i) for i in range(0, 3)]
hB_set = [2**(7+i) for i in range(0, 6)]
w_set = [2**(7+i) for i in range(0,6)]
transpose = [0, 1]

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

            for hA_c in hA_set:
                for hB_c in hB_set:
                    for w_c in w_set:
                        for tr_c in transpose:
                            arg = arg_template % (hA_c, hB_c, w_c, tr_c)
                            print 'current arg: ', arg

                            # start record power data
                            os.system('echo sm_clock:%s,mem_clock:%s >> %s%s_%s_%s_caffe_dvfs.txt' % (core_f, mem_f, pw_log_folder, hA_c, hB_c, w_c))
                            os.system('echo program:%s,arguments:%s >> %s%s_%s_%s_caffe_dvfs.txt' % (app, arg, pw_log_folder, hA_c, hB_c, w_c))
                            command = 'start /B nvidia-smi.exe -l 1 >> %s%s_%s_%s_caffe_dvfs.txt' % (pw_log_folder, hA_c, hB_c, w_c)
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
                            # time.sleep(wt)

# reset GPU first
command = 'nvidiaInspector.exe -forcepstate:%s,16' % nvIns_dev_id
print command
os.system(command)
time.sleep(wt)
os.system('pause')