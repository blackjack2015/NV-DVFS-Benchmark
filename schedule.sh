gpu=gtx2070s-dvfs
kernels=real
os=windows
trace=trace3

python scheduler.py --benchmark-setting ${gpu} --kernel-setting ${kernels} --app-root applications/windows/cuda11 \
	            --trace $trace
