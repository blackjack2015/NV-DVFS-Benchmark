import time, argparse
import datetime
import sys
import csv
import os, glob, re
import numpy as np
import json
import pandas as pd
import re


def parse_perf_log(perf_file):
    fn = perf_file.split('/')[-1]
    print(fn)

    baseInfo = fn.split('_')
    appName = baseInfo[1]
    coreF = str(int(baseInfo[2][4:]))
    memF = str(int(baseInfo[3][3:]))
    argNo = baseInfo[4]

    rec = [appName, coreF, memF, argNo]

    # extract execution time information
    time = None
    f = open(perf_file, 'r')
    content = f.readlines()
    f.close()

    # extract the kernel name
    kernel = content[0].split(':')[1].strip()

    isLog = True
    if isLog:
        regex = re.compile(r'(iterated \d+, average time is)|(Average Kernel Time)|(Average Time)')
        timeRaw = filter(regex.search, content)
        timeRaw = list(timeRaw)
        if len(timeRaw) != 0:
            time = float(timeRaw[-1].split()[-2].strip())
    else:  
        regex = re.compile(r'.*\%.*' + kernel)
        time = filter(regex.search, content)[0]
        time = time.replace("GPU activities:", "")
        time = time.split()[3].strip()
        if 'us' in time:
            time = float(time[:-2]) / 1000
        elif 'ms' in time:
            time = float(time[:-2])
        else:
            time = float(time[:-1]) * 1000

    dict_info = {
        'benchmark': appName,
        'argNo': argNo,
        'kernel': kernel,
        'core_frequency': coreF,
        'memory_frequency': memF,
        'time': time
    }

    return dict_info


def parse_metrics_log(metrics_file):
    # extract grid and block settings
    f = open(metrics_file, 'r')
    content = f.readlines()
    maxLen = len(content)
    f.close()
    regex = re.compile(r'\(\d+ \d+ \d+\).*(void )*' + kernel + r'[<(]+')
    message = [value \
                for s, value in enumerate(content) \
                    for m in [regex.search(value)] if m]
    if len(message) != 0:
        message = message[0]
        # print message
        grid_block = re.findall(r'\(\d+ \d+ \d+\)', message)
        grid_block = " ".join(grid_block)
        warps_info = [int(item) for item in grid_block.translate(None, '()').split()]
        warps = float(np.prod(np.array(warps_info))) / 32.0
        print(grid_block, warps)
        rec.append(grid_block)
        rec.append(warps)
    else:
        rec.append(" ")
        rec.append(" ")

    # extract metrics information
    stride = 2
    fm, number = re.subn('perf', 'metrics', fp)
    f = open(fm, 'r')
    content = f.readlines()
    maxLen = len(content)
    f.close()
    regex = re.compile(r'Kernel: (void )*' + kernel + r'[<(]+')
    message = [line.strip() \
                for s, value in enumerate(content) \
                    for m in [regex.search(value)] if m \
                for line in content[s+1:min(maxLen,s+1+stride)]]

    print(len(message))
    if len(message) == 46:
        message = message[:-1]

    for line in message:
        metric = line.split()[1]

        # initialize metrics list
        if len(metrics) < len(message):
            metrics.append(metric)

        value = line.split()[-1]

        if '%' in value:
            value = float(value[:-1]) / 100
        elif 'TB/s' in value:
            value = float(value[:-4]) * 1e3
        elif 'GB/s' in value:
            value = float(value[:-4])
        elif 'MB/s' in value:
            value = float(value[:-4]) * 1e-3
        elif 'KB/s' in value:
            value = float(value[:-4]) * 1e-6
        elif 'B/s' in value:
                value = float(value[:-3]) * 1e-9

        rec.append(value)


def parse_power_log(power_file):
    baseInfo = power_file.split('_')
    appName = baseInfo[1]
    coreF = int(baseInfo[2][4:])
    memF = int(baseInfo[3][3:])
    argNo = baseInfo[4]

    # neglect first two lines of device information and table header
    f = open(power_file, 'r')
    content = f.readlines()[2:]
    f.close()
    
    powerList = [float(line.split()[-1].strip()) / 1000.0 for line in content if int(line.split()[3]) == coreF]

    # powerList = [float(line.split()[-1].strip()) / 1000.0 for line in content if int(line.split()[1]) == runState]

    #powerList = powerList[len(powerList) / 10 * 5 :len(powerList) / 10 * 6]   # filter out those power data of cooling down GPU
    powerList.sort()
    powerList = powerList[-10:]   # filter out those power data of cooling down GPU
    avg_power = np.mean(powerList)

    dict_info = {
        'average_power': avg_power
    }

    return dict_info


def parse_dcgm_log(dcgm_file):
    baseInfo = dcgm_file.split('_')
    appName = baseInfo[1]
    coreF = int(baseInfo[2][4:])
    memF = int(baseInfo[3][3:])
    argNo = baseInfo[4]

    # neglect first two lines of device information and table header
    f = open(dcgm_file, 'r')
    content = f.readlines()
    f.close()

    metrics = content[0].strip()
    metrics = re.split(r"[ ]+", metrics)[1:]
    lines = [line for line in content if 'GPU' in line]
    lines = [line for line in lines if 'N/A' not in line]
    data = [re.split(r"[ ]+", line.strip())[2:] for line in lines]
    data = np.array(data, dtype=np.float)
    data = data[data[:, 0] > 0]   # filter data of which SMACT > 0
    means = np.mean(data, axis=0).tolist()

    dict_info = {}
    for i in range(len(metrics)):
        dict_info[metrics[i]] = means[i]

    return dict_info

