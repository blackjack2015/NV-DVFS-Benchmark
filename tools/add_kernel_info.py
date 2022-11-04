import glob
import configparser
import json


logRoot = 'logs/v100-dvfs-real'
perf_filelist = glob.glob(r'%s/*perf.log' % logRoot)

cf_ks = configparser.ConfigParser()
cf_ks.read('configs/kernels/real.cfg')

for perf_file in perf_filelist:

    appInfo = perf_file.split('_')
    appName = appInfo[1]
    argNo = int(appInfo[4][5:])

    kernel = json.loads(cf_ks.get(appName, 'kernels'))[argNo]
    print(appName, kernel)

    with open(perf_file, 'r+') as f:
        content = f.read()
        f.seek(0, 0)
        f.write("kernel_name:%s\n" % kernel + content)
