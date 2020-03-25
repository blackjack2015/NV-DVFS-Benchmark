gpu=v100-dvfs
kernels=real
os=linux
perf_algo=qiang2018
power_algo=xgboost

# run benchmarks
#python dvfs_benchmark.py --benchmark-setting ${gpu} --kernel-setting ${kernels} --app-root applications/${os}

# extract performance data
#python gpuPerfExtractor.py --benchmark-setting ${gpu} --kernel-setting ${kernels}

# use analytical model to fit performance data
python analytical.py --data-root raw --benchmark-setting ${gpu} --kernel-setting ${kernels} --method ${perf_algo}

# use ML model to fit power data
#python power_dvfs.py --benchmark-setting ${gpu} --kernel-setting ${kernels} --method${power_algo}

