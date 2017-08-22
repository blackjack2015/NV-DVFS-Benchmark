import time
import datetime
import sys,urllib,urllib2
import csv
import os
import cPickle as pickle
# import MySQLdb
import scipy.io as sio
import numpy as np

# prepare csv file
#csvfile = open('DVFS-Performance.csv', 'wb')
#csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
#basicHeader = ['Application', 'kernel', 'sm_clock', 'mem_clock', 'time']
#csvWriter.writerow(basicHeader + metrics)

logRoot = 'EPPMiner_DVFS_gtx980/'

# Read Configuration
fileName = [line.strip() for line in open('logList.txt').readlines()]
profKW = [line.strip() for line in open('kernelList.txt').readlines()]
metrics = [line.strip() for line in open('metricsList.txt').readlines()]
freqKW = 'sm_clock'
argKW = 'arguments:'
outTxt = open('perfStatistics.txt', 'w')

# Initial database connection
#conn_localhost=MySQLdb.connect(host='127.0.0.1',user='root',passwd='root',db='gpu_dvfs',port=3306)
#cur=conn_localhost.cursor()

record = [];
for i in range(len(profKW)):
    try:
        myfile = open(logRoot + 'performance/' + fileName[i] + '.txt');
    except Exception, e:
        print e.args[0], e.args[1]
        continue;

    rawText = myfile.readlines()

    freqInd = [s for s, line in enumerate(rawText) if freqKW in line];
    freqInfo = [rawText[l] for l in freqInd];
    print len(freqInfo);

    argInd = [s for s, line in enumerate(rawText) if argKW in line];
    argInfo = [rawText[l] for l in argInd];
    print len(argInfo);

    profInd = [s for s, line in enumerate(rawText) if profKW[i] in line];
    profTime = [rawText[l] for l in profInd];
    print len(profTime)

    offset = 1;
    for j in range(len(freqInfo)):
        dataArray = [];
        # print j

        # Extract kernel name
        dataArray.append(fileName[i])
        dataArray.append(profKW[i])

        # Extract clock information
        freqRecord = freqInfo[j];
        sm_clock = int(freqRecord[freqRecord.find("sm_clock:") + len("sm_clock:"):freqRecord.find(",mem_clock:")].strip())
        mem_clock = int(freqRecord[freqRecord.find(",mem_clock:") + len(",mem_clock:"):].strip())
        dataArray.append(sm_clock)
        dataArray.append(mem_clock)

        # Extract arguments information
        argRecord = argInfo[j];
        arg = argRecord[argRecord.find("arguments:") + len("arguments:"):].strip()
        dataArray.append(arg)

        # Extract running time
        profTRecord = rawText[profInd[j * 2]];
        prof_time = profTRecord.split()[3]
        if prof_time.endswith("ms"):
            prof_time = float(prof_time.replace('ms', ''));
        elif prof_time.endswith("us"):
            prof_time = float(prof_time.replace('us', '')) / 1.0e3;
        elif prof_time.endswith("s"):
            prof_time = float(prof_time.replace('s', '')) * 1.0e3;  
        dataArray.append(prof_time)

        # Extract profiling information
        profRecord = rawText[profInd[j * 2 + 1] + 1 : profInd[j * 2 + 1] + len(metrics) + 1];
        metricsRecs = [line.split()[-1] for line in profRecord];
        for k in range(0, len(metricsRecs)):
            if metricsRecs[k].endswith("GB/s"):
                metricRecss[k] = float(metricsRecs[k].replace('GB/s', ''))
            elif metricsRecs[k].endswith("MB/s"):
                metricsRecs[k] = float(metricsRecs[k].replace('MB/s', '')) / 1.0e3
            elif metricsRecs[k].endswith("KB/s"):
                metricsRecs[k] = float(metricsRecs[k].replace('KB/s', '')) / 1.0e6  
            elif metricsRecs[k].endswith("B/s"):
                metricsRecs[k] = float(metricsRecs[k].replace('B/s', '')) / 1.0e9
            elif metricsRecs[k].endswith("%"):
		metricsRecs[k] = float(metricsRecs[k][:-2]) / 100.0
	    else:
                metricsRecs[k] = float(metricsRecs[k]);
            # metricsRecs[k] = str(metricsRecs[k])
        dataArray = dataArray + metricsRecs

        recStr = '|'
        for item in dataArray:
            recStr += str(item) + '|'
        
        recStr += '\n'
        print recStr
        outTxt.write(recStr)

        record.append(dataArray)
        #cur.execute('insert into gpu_performance_dvfs(App, Kernel,\
        #                    SM_Clock, MEM_Clock,\
        #                    Arguments, Time,\
        #                    Achieved_Occ,\
        #                    Dram_read_throughput, Dram_read_transactions,\
        #                    Dram_write_throughput, Dram_write_transactions,\
        #                    l2_read_throughput, l2_read_transactions,\
        #                    shared_load_throughput, shared_load_transactions,\
        #                    shared_store_throughput, shared_store_transactions\
        #                    ) values(%s, %s, %s, %s, %s, %s, %s, \
        #                             %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)',\
        #                    dataArray)
        
        #cur.execute('insert into gpu_performance_dvfs(App, Kernel,\
        #                    SM_Clock, MEM_Clock,\
        #                    Arguments, Time\
        #                    ) values(%s, %s, %s, %s, %s, %s)',\
        #                    dataArray)

        ##write a row to csv
        #csvWriter.writerow(dataArray)

# print record
# sio.savemat('perfData.mat', {'perfData':np.array(record)})
outTxt.close()
tempf = open('perfData.bin', 'wb')
pickle.dump(record, tempf, 0)
tempf.close()

#conn_localhost.commit()
#cur.close()
#conn_localhost.close()

﻿import time
import datetime
import sys,urllib,urllib2
import csv
import os
import cPickle as pickle
# import MySQLdb
import scipy.io as sio
import numpy as np

# prepare csv file
#csvfile = open('DVFS-Performance.csv', 'wb')
#csvWriter = csv.writer(csvfile, dialect='excel')

# write table head
#basicHeader = ['Application', 'kernel', 'sm_clock', 'mem_clock', 'time']
#csvWriter.writerow(basicHeader + metrics)

logRoot = 'EPPMiner_DVFS_gtx980/'

# Read Configuration
fileName = [line.strip() for line in open('logList.txt').readlines()]
profKW = [line.strip() for line in open('kernelList.txt').readlines()]
metrics = [line.strip() for line in open('metricsList.txt').readlines()]
freqKW = 'sm_clock'
argKW = 'arguments:'
outTxt = open('perfStatistics.txt', 'w')

# Initial database connection
#conn_localhost=MySQLdb.connect(host='127.0.0.1',user='root',passwd='root',db='gpu_dvfs',port=3306)
#cur=conn_localhost.cursor()

record = [];
for i in range(len(profKW)):
    try:
        myfile = open(logRoot + 'performance/' + fileName[i] + '.txt');
    except Exception, e:
        print e.args[0], e.args[1]
        continue;

    rawText = myfile.readlines()

    freqInd = [s for s, line in enumerate(rawText) if freqKW in line];
    freqInfo = [rawText[l] for l in freqInd];
    print len(freqInfo);

    argInd = [s for s, line in enumerate(rawText) if argKW in line];
    argInfo = [rawText[l] for l in argInd];
    print len(argInfo);

    profInd = [s for s, line in enumerate(rawText) if profKW[i] in line];
    profTime = [rawText[l] for l in profInd];
    print len(profTime)

    offset = 1;
    for j in range(len(freqInfo)):
        dataArray = [];
        # print j

        # Extract kernel name
        dataArray.append(fileName[i])
        dataArray.append(profKW[i])

        # Extract clock information
        freqRecord = freqInfo[j];
        sm_clock = int(freqRecord[freqRecord.find("sm_clock:") + len("sm_clock:"):freqRecord.find(",mem_clock:")].strip())
        mem_clock = int(freqRecord[freqRecord.find(",mem_clock:") + len(",mem_clock:"):].strip())
        dataArray.append(sm_clock)
        dataArray.append(mem_clock)

        # Extract arguments information
        argRecord = argInfo[j];
        arg = argRecord[argRecord.find("arguments:") + len("arguments:"):].strip()
        dataArray.append(arg)

        # Extract running time
        profTRecord = rawText[profInd[j * 2]];
        prof_time = profTRecord.split()[3]
        if prof_time.endswith("ms"):
            prof_time = float(prof_time.replace('ms', ''));
        elif prof_time.endswith("us"):
            prof_time = float(prof_time.replace('us', '')) / 1.0e3;
        elif prof_time.endswith("s"):
            prof_time = float(prof_time.replace('s', '')) * 1.0e3;  
        dataArray.append(prof_time)

        # Extract profiling information
        profRecord = rawText[profInd[j * 2 + 1] + 1 : profInd[j * 2 + 1] + len(metrics) + 1];
        metricsRecs = [line.split()[-1] for line in profRecord];
        for k in range(0, len(metricsRecs)):
            if metricsRecs[k].endswith("GB/s"):
                metricRecss[k] = float(metricsRecs[k].replace('GB/s', ''))
            elif metricsRecs[k].endswith("MB/s"):
                metricsRecs[k] = float(metricsRecs[k].replace('MB/s', '')) / 1.0e3
            elif metricsRecs[k].endswith("KB/s"):
                metricsRecs[k] = float(metricsRecs[k].replace('KB/s', '')) / 1.0e6  
            elif metricsRecs[k].endswith("B/s"):
                metricsRecs[k] = float(metricsRecs[k].replace('B/s', '')) / 1.0e9
            elif metricsRecs[