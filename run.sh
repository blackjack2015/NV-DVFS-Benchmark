# run benchmarks
#python dvfs_benchmark.py --benchmark-setting p100 --kernel-setting synthetic --app-root applications/linux
#python dvfs_benchmark.py --benchmark-setting p100 --kernel-setting real --app-root applications/linux
#python dvfs_benchmark.py --benchmark-setting p100-dvfs --kernel-setting real --app-root applications/linux
#python dvfs_benchmark.py --benchmark-setting titanx --kernel-setting spgemm --app-root applications/linux

# use analytical model to fit performance data
python analytical.py --data-root raw --benchmark-setting gtx980-low-dvfs --kernel-setting real-small-workload --method qiang2018
python analytical.py --data-root raw --benchmark-setting gtx1080ti-dvfs --kernel-setting real --method qiang2018
python analytical.py --data-root raw --benchmark-setting p100-dvfs --kernel-setting real --method qiang2018

## use ML model to fit power data
#python power_dvfs.py --benchmark-setting gtx980-low-dvfs --kernel-setting real-small-workload --method svr-poly
#python power_dvfs.py --benchmark-setting gtx980-low-dvfs --kernel-setting real-small-workload --method svr-rbf
#python power_dvfs.py --benchmark-setting gtx980-low-dvfs --kernel-setting real-small-workload --method xgboost
##python power_dvfs.py --benchmark-setting gtx980-high-dvfs --kernel-setting real-small-workload --method svr-poly
##python power_dvfs.py --benchmark-setting gtx980-high-dvfs --kernel-setting real-small-workload --method svr-rbf
##python power_dvfs.py --benchmark-setting gtx980-high-dvfs --kernel-setting real-small-workload --method xgboost
#python power_dvfs.py --benchmark-setting gtx1080ti-dvfs --kernel-setting real --method svr-poly
#python power_dvfs.py --benchmark-setting gtx1080ti-dvfs --kernel-setting real --method svr-rbf
#python power_dvfs.py --benchmark-setting gtx1080ti-dvfs --kernel-setting real --method xgboost
#python power_dvfs.py --benchmark-setting p100-dvfs --kernel-setting real --method svr-poly
#python power_dvfs.py --benchmark-setting p100-dvfs --kernel-setting real --method svr-rbf
#python power_dvfs.py --benchmark-setting p100-dvfs --kernel-setting real --method xgboost
