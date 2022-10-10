import os
import sys
import argparse
import subprocess
import time
import re
import configparser
import json
from utils.profiler import PowerProfiler
from utils.profiler import NvProfiler
from utils.profiler import DCGMProfiler
from utils.dvfs_control import DVFSController


class Benchmark:

    def __init__(self, app_dir, log_dir, application, arg_no, arg, core_freq, mem_freq):

        # app_exec_cmd = './%s/%s %s -device=%d -secs=%d >> %s/%s' % (
        self.base_cmd = './%s/%s %s' % (
            app_dir,
            application,
            arg
        )

        # arg, number = re.subn('-device=[0-9]*', '-device=%d' % cuda_dev_id, arg)
        self.powerlog = './%s/benchmark_%s_core%d_mem%d_input%03d_power.log' % (log_dir, application, core_freq, mem_freq, arg_no)
        self.perflog = './%s/benchmark_%s_core%d_mem%d_input%03d_perf.log' % (log_dir, application, core_freq, mem_freq, arg_no)
        self.metricslog = './%s/benchmark_%s_core%d_mem%d_input%03d_metrics.log' % (log_dir, application, core_freq, mem_freq, arg_no)
        self.dcgmlog = './%s/benchmark_%s_core%d_mem%d_input%03d_dcgm.log' % (log_dir, application, core_freq, mem_freq, arg_no)

    def get_power_file(self):

        return self.powerlog

    def get_performance_file(self):

        return self.perflog

    def get_metrics_file(self):

        return self.metricslog

    def get_dcgm_file(self):

        return self.dcgmlog

    def get_run_command(self, device_id=0, iters=100, secs=None):

        if secs is None:
            command = '%s -device=%d -iters=%d' % (self.base_cmd, device_id, iters)
        else:
            command = '%s -device=%d -secs=%d' % (self.base_cmd, device_id, secs)
        return command

    def run(self, device_id=0, iters=100, secs=None):

        command = self.get_run_command(device_id, iters=iters, secs=secs)
        command += ' 1>>%s 2>&1' % self.perflog
        print(command)
        os.system(command)


def get_config(bench_file):

    BS_SETTING = '%s.cfg' % bench_file

    bench_args = {}
    
    # Reading benchmark settings
    cf_bs = configparser.ConfigParser()
    cf_bs.read("configs/benchmarks/%s" % BS_SETTING)

    # device info
    bench_args['cuda_dev_id'] = cf_bs.getint("device", "cuda_device_id")
    bench_args['nvins_dev_id'] = cf_bs.getint("device", "nvins_device_id")
    bench_args['nvsmi_dev_id'] = cf_bs.getint("device", "nvsmi_device_id")

    # global running config
    bench_args['running_time'] = cf_bs.getint("global", "secs")
    bench_args['rest_time'] = cf_bs.getint("global", "rest_time")
    bench_args['pw_sample_int'] = cf_bs.getint("global", "power_sample_interval")

    # nvprof
    bench_args['nvprof_time_cmd'] = cf_bs.get("nvprof", "time_command")
    bench_args['nvprof_thread_cmd'] = cf_bs.get("nvprof", "setting_command")
    bench_args['nvprof_metrics_cmd'] = cf_bs.get("nvprof", "metrics_command")
    bench_args['nvprof_metrics_list'] = json.loads(cf_bs.get("nvprof", "metrics"))

    # dvfs control
    bench_args['core_freqs'] = json.loads(cf_bs.get("dvfs_control", "coreF"))
    bench_args['mem_freqs'] = json.loads(cf_bs.get("dvfs_control", "memF"))
    bench_args['freqs'] = [(coreF, memF) for coreF in bench_args['core_freqs'] for memF in bench_args['mem_freqs']]

    return bench_args


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-setting', type=str, help='gpu benchmark setting', default='v100-dvfs')
    parser.add_argument('--kernel-setting', type=str, help='kernels of benchmark', default='matrixMul')
    parser.add_argument('--app-root', type=str, help='folder of applications', default='applications/linux')
    
    opt = parser.parse_args()
    print(opt)
    
    application_dir = opt.app_root
    logging_dir = 'logs/%s-%s' % (opt.benchmark_setting, opt.kernel_setting)
    
    try:
        os.makedirs(logging_dir)
    except OSError:
        pass
    
    bench_args = get_config(opt.benchmark_setting)
    
    # Read GPU application settings
    KS_SETTING = '%s.cfg' % opt.kernel_setting
    cf_ks = configparser.ConfigParser()
    cf_ks.read("configs/kernels/%s" % KS_SETTING)
    benchmark_programs = cf_ks.sections()
    
    power_profiler = PowerProfiler(
        device_id = bench_args['nvsmi_dev_id'],
        sample_interval = bench_args['pw_sample_int']
    )
    nvprofiler = NvProfiler(device_id=bench_args['cuda_dev_id'], metrics=bench_args['nvprof_metrics_list'])
    dcgm_profiler = DCGMProfiler(device_id=bench_args['nvsmi_dev_id'])
    dvfs_controller = DVFSController(device_id=bench_args['nvsmi_dev_id'])

    for core_freq, mem_freq in bench_args['freqs']:
    
        # set specific frequency
        dvfs_controller.set_frequency(core_freq, mem_freq)
        
        for i, app in enumerate(benchmark_programs):
    
            args = json.loads(cf_ks.get(app, 'args'))
    
            for argNo, arg in enumerate(args):
    
                bench = Benchmark(
                    app_dir = application_dir,
                    log_dir = logging_dir,
                    application = app,
                    arg_no = argNo,
                    arg = arg,
                    core_freq = core_freq,
                    mem_freq = mem_freq
                )
    
                # start record power data
                power_profiler.start(bench.get_power_file())
                time.sleep(bench_args['rest_time'])
    
                # execute program to collect power data
                bench.run(secs=bench_args['running_time'])
                time.sleep(bench_args['rest_time'])
    
                # stop record power data
                power_profiler.end()
    
                # use nvprof to collect the execution time
                nvprofiler.collect_time(bench)
                time.sleep(bench_args['rest_time'])
                
                # use nvprof to collect the thread setting
                nvprofiler.collect_thread_setting(bench)
                time.sleep(bench_args['rest_time'])
                
                # use nvprof to collect the metrics
                nvprofiler.collect_metrics(bench)
                time.sleep(bench_args['rest_time'])
                

