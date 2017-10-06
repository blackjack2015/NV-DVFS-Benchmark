import time
import datetime
import sys,urllib,urllib2
import csv
import os, glob, re
import cPickle as pickle
# import MySQLdb
# import scipy.io as sio
import numpy as np
import ConfigParser
import json
import pandas as pd

logRoot = 'logs'

perf_filelist = glob.glob(r'%s/*perf.log' % logRoot)
metrics_filelist = glob.glob(r'%s/*metrics.log' % logRoot)

perf_filelist.sort()
metrics_filelist.sort()

# Reading metrics list
cf_bs = ConfigParser.SafeConfigParser()
cf_bs.read("configs/benchmark_settings.cfg")
metrics = json.loads(cf_bs.get("profile_control", "metrics"))

# Read GPU application settings
cf_ks = ConfigParser.SafeConfigParser()
cf_ks.read("configs/kernels_settings.cfg")
benchmark_programs = cf_ks.sections()

head = ["appName", "coreF", "memF", "argNo", "kernel", "time"] + metrics
print head

# prepare csv file
csvfile = open('DVFS-Performance.csv', 'wb')
csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
csvWriter.writerow(head)

for fp in perf_filelist:
    # print fp

    baseInfo = fp.split('_')
    appName = baseInfo[1]
    coreF = baseInfo[2][4:]
    memF = baseInfo[3][3:]
    argNo = baseInfo[4]

    kernel = json.loads(cf_ks.get(appName, 'kernels'))[0]
    rec = [appName, coreF, memF, argNo, kernel]

    # extract execution time information
    f = open(fp, 'r')
    content = f.readlines()
    f.close()
    regex = re.compile(r'.*\%.*' + kernel)
    time = filter(regex.search, content)[0].split()[3].strip()
    rec.append(time)

    # extract metrics information
    fm, number = re.subn('perf', 'metrics', fp)
    f = open(fm, 'r')
    content = f.readlines()
    f.close()
    message = [line.strip() \
                for s, value in enumerate(content) \
                    if 'Kernel: %s' % kernel in value or 'Kernel: void %s' % kernel in value \
                for line in content[s+1:s+4]]

    for line in message:
        rec.append(line.split()[-1])

    # print rec
    csvWriter.writerow(rec)


# tempf = open('perfData.bin', 'wb')
# pickle.dump(record, tempf, 0)
# tempf.close()
