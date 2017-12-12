import time
import datetime
import sys,urllib,urllib2
import csv
import os, glob, re
import cPickle as pickle
import numpy as np
import ConfigParser
import json
import pandas as pd

logRoot = 'ipdps 2018 power logs'

power_filelist = glob.glob(r'%s/*power.log' % logRoot)
power_filelist.sort()

# Reading metrics list
cf_bs = ConfigParser.SafeConfigParser()
cf_bs.read("benchmark_settings.cfg")
metrics = json.loads(cf_bs.get("profile_control", "metrics"))

# Read GPU application settings
cf_ks = ConfigParser.SafeConfigParser()
cf_ks.read("kernels_settings.cfg")
benchmark_programs = cf_ks.sections()

head = ["appName", "coreF", "memF", "argNo", "kernel", "power/W"]
print head

# prepare csv file
csvfile = open('DVFS-Power.csv', 'wb')
csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
csvWriter.writerow(head)

for fp in power_filelist:
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
    content = f.readlines()[2:]
    f.close()
    powerList = [float(line.split()[3].strip()) / 1000.0 for line in content if line.split()[0] == '5']
    powerList = powerList[len(powerList) / 3:len(powerList) * 2 / 3]   # filter out those power data of cooling down GPU
    rec.append(np.mean(powerList))

    # print rec
    csvWriter.writerow(rec[:len(head)])

