import glob
import configparser
import json


logRoot = 'logs/v100-dvfs-microbenchmark'
perf_filelist = glob.glob(r'%s/*perf.log' % logRoot)

cf_ks = configparser.ConfigParser()
cf_ks.read('configs/kernels/microbenchmark.cfg')
benchmark_programs = cf_ks.sections()
print(benchmark_programs)

for perf_file in perf_filelist:

    appInfo = perf_file.split('_')
    appName = appInfo[1]
    argNo = int(appInfo[4][5:])

    print(appName, argNo)
    kernel = json.loads(cf_ks.get(appName, 'kernels'))[argNo]
    print(appName, kernel)
