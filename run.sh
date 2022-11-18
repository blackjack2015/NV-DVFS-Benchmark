GPU=rtx8000-dvfs

python tools/gpuParse.py --benchmark-setting ${GPU} --kernel-setting real
python tools/gpuParse.py --benchmark-setting ${GPU} --kernel-setting microbenchmark

python tools/dataMerger.py --profile-csv data/${GPU}-real-profile.csv --output data/${GPU}-real.csv
python tools/dataMerger.py --profile-csv data/${GPU}-microbenchmark-profile.csv --output data/${GPU}-microbenchmark.csv

