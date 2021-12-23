gpu=gtx2070s
kernels=real
os=windows

python dvfs_benchmark.py --benchmark-setting ${gpu} --kernel-setting ${kernels} --app-root applications/windows/cuda11
