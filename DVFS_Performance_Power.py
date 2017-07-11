import os
import subprocess
import time
import re

app_folder = 'Applications/'
log_folder = 'Logs/performance_power/'
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