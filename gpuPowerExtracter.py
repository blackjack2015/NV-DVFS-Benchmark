k].endswith("%"):
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
import re
import cPickle as pickle

logRoot = 'EPPMiner_DVFS_gtx980/'

# Read log lines
# loglines = [line.strip() for line in open('dvfs.txt').readlines()]
outTxt = open('powerStatistics.txt', 'w')

fileName = [line.strip() for line in open('logList.txt').readlines()]

loglines = []
for i in range(len(fileName)):
    try:
        myfile = open(logRoot + 'power/' + fileName[i] + '.txt')
	tmplines = [line.strip() for line in myfile.readlines()]
	loglines.extend(tmplines)
    except Exception, e:
        print e.args[0], e.args[1], fileName
        continue;

# for line in loglines:
#     print line

freqInd = [s for s, line in enumerate(loglines) if 'sm_clock:' in line]
# progInd = [s for s, line in enumerate(loglines) if 'program:' in line];
record_count = len(freqInd)

outRec = []
for i in range(record_count):

    if i < record_count - 1:
        powerSamples = loglines[freqInd[i] : freqInd[i+1]]
    else:
        powerSamples = loglines[freqInd[i] : ]

    # extract clock frequency and program information
    clock_info = powerSamples[0]
    sm_clock = re.findall('sm_clock:[0-9]*,', clock_info)[0][9:-1]
    mem_clock = re.findall('mem_clock:[0-9]*', clock_info)[0][10:]
    prog_info = powerSamples[1]
    program = re.findall('program:[A-Za-z|(|)|-]*,', prog_info)[0][8:-1]
    arg = re.findall('arguments:.*', prog_info)[0][10:]
    powerSamples = powerSamples[2:]

    powerRecInd = [s for s, line in enumerate(powerSamples) if 'Driver Version:' in line]
    tmp_rec_count = len(powerRecInd)
    record = [powerSamples[powerRecInd[i]:powerRecInd[i + 1]] if i < tmp_rec_count - 1 \
              else powerSamples[powerRecInd[i]:] for i in range(tmp_rec_count)]

    idlePowerInfor = []
    actPowerInfor = []
    # tempInfor = []
    appInfor = []

    for infor in record:
        powerLine = [infor[j + 1] for j, line in enumerate(infor) if 'GTX 980' in line];
        if len(powerLine) == 0:         # incomplete nvidia-smi returned records
            continue
        else:
            power = powerLine[0].split()[4];
        # print power;

        #tempLine = [infor[j + 1] for j, line in enumerate(infor) if 'GTX 980' in line][0];
        #temp = powerLine.split()[2];
        ## print temp;

        appLine = [infor[k] for k, line in enumerate(infor) if 'Applications' in line];
        if len(appLine) == 0:
            app = ' '
            appInfor.append(app)
            idlePowerInfor.append(power)
        else:
            app = appLine[0].split()[4]
            app = app.split('\\')[-1]
            appInfor.append(app)
            actPowerInfor.append(power)

        print app, power

    # find most frequent active power
    act_p = 0
    idle_p = 0
    max_count = -1
    tmp_set = set(actPowerInfor)
    for p in tmp_set:
        if actPowerInfor.count(p) > max_count:
            max_count = actPowerInfor.count(p)
            act_p = p
    if act_p == 0 or act_p == '0':
        continue
    max_count = -1
    tmp_set = set(idlePowerInfor)
    for p in tmp_set:
        if idlePowerInfor.count(p) > max_count:
            max_count = idlePowerInfor.count(p)
            idle_p = p

    outRec.append([int(sm_clock), int(mem_clock), program, arg, int(str(idle_p).replace('W', '')), int(str(act_p).replace('W', ''))])
    txt_rec = '%s|%s|%s|%s|%s|%s\n' % (int(sm_clock), int(mem_clock), program, arg, idle_p, act_p)
    # print txt_rec
    outTxt.write(txt_rec)

outTxt.close()
tempf = open('powerData.bin', 'wb')
pickle.dump(outRec, tempf, 0)
tempf.close()
