import os


def run_command(command):

    print(command)
    os.system(command)


class PowerProfiler:

    def __init__(self, device_id, sample_interval=200):

        self.device_id = device_id
        self.sample_interval = sample_interval

        self.start_cmd = './tools/nvml_tools -device=%d -si=%d' % (
            self.device_id,
            self.sample_interval,
        )
        self.stop_cmd = 'killall nvml_tools'

        # for windows commands, to be discarded
        # pw_sampling_cmd = 'start /B nvml_samples.exe -device=%d -si=%d -output=%s/%s > nul'
        # kill_pw_cmd = 'tasklist|findstr "nvml_samples.exe" && taskkill /F /IM nvml_samples.exe'
    def start(self, output):

        daemon_cmd = 'nohup %s 1>%s 2>&1 &' % (self.start_cmd, output)
        run_command(daemon_cmd)

    def end(self):

        run_command(self.stop_cmd)

    def get_current_val(self):

        pass


class NvProfiler:

    def __init__(self, device_id=0, metrics=None):

        self.device_id = device_id
        self.metrics = metrics

    def collect_time(self, bench):

        # execute program to collect time data
        bench_cmd = bench.get_run_command(device_id=self.device_id, secs=5)
        command = 'nvprof --profile-child-processes %s 1>>%s 2>&1' % (bench_cmd, bench.get_performance_file())
        run_command(command)

    def collect_thread_setting(self, bench):

        # collect grid and block settings
        bench_cmd = bench.get_run_command(device_id=self.device_id, iters=10)
        command = 'nvprof --print-gpu-trace --profile-child-processes %s 1>>%s 2>&1' % (bench_cmd, bench.get_metrics_file())
        run_command(command)

    def collect_metrics(self, bench):

        # execute program to collect metrics data
        bench_cmd = bench.get_run_command(device_id=self.device_id, iters=10)
        metCount = 0

        # the metric number cannot be too large due to memory limit
        metrics = self.metrics
        stride = 2
        while metCount < len(metrics):

            if metCount + stride > len(metrics):
                metStr = ','.join(metrics[metCount:])
            else:
                metStr = ','.join(metrics[metCount:metCount + stride])
            command = 'nvprof --devices %s --metrics %s %s 1>>%s 2>&1' % (self.device_id, metStr, bench_cmd, bench.get_metrics_file())
            run_command(command)
            metCount += stride


class DCGMProfiler:

    def __init__(self, device_id=0, sample_interval=200):

        self.device_id = device_id
        self.sample_interval = sample_interval
        self.start_cmd = 'dcgmi dmon -i %d -d %d' % (
            self.device_id,
            self.sample_interval,
        )
        self.stop_cmd = 'killall dcgmi'
        # SMACT - 1002
        # SMOCC - 1003
        # TENSO - 1004
        # DRAMA - 1005
        # FP64A - 1006
        # FP32A - 1007
        # FP16A - 1008
        self.metrics = [1002, 1003, 1004, 1005, 1006, 1007, 1008]

    def start(self, output):

        metrics_str = '-e ' + ','.join([str(item) for item in self.metrics])
        daemon_cmd = 'nohup %s %s 1>%s 2>&1 &' % (self.start_cmd, metrics_str, output)
        run_command(daemon_cmd)

    def end(self):

        run_command(self.stop_cmd)

