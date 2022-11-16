python tools/gpuParse.py --kernel-setting real
python tools/gpuParse.py --kernel-setting microbenchmark

python tools/dataMerge.py --profile-csv data/v100-dvfs-real-profile.csv --output data/v100-dvfs-real.csv
python tools/dataMerge.py --profile-csv data/v100-dvfs-microbenchmark-profile.csv --output data/v100-dvfs-microbenchmark.csv

