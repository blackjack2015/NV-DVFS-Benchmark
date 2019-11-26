import pandas as pd
import numpy as np
import sys,os
import random
from settings import *

def energy_best(gpu, version):
    
    csv_file = "csvs/analytical/results/%s-%s-qiang2018-dvfs.csv" % (gpu, version)
    perf_data = pd.read_csv(csv_file, header = 0)
    csv_file = "csvs/ml/%s-%s-xgboost-Power.csv" % (gpu, version)
    pow_data = pd.read_csv(csv_file, header = 0)

    if gpu == 'gtx980-low-dvfs':
        GPUCONF = GTX980('low')
    elif gpu == 'gtx1080ti-dvfs':
        GPUCONF = GTX1080TI()
    elif gpu == 'p100-dvfs':
        GPUCONF = P100()

    energy_data = pd.DataFrame([])

    kernelset = perf_data['kernel'].drop_duplicates().reset_index(drop=True)
    #print kernelset

    energy_data['appName'] = kernelset
    energy_data['defaultE'] = None
    energy_data['bestE'] = None
    energy_data['bestC'] = None
    energy_data['bestM'] = None
    energy_data['predictE'] = None
    energy_data['predictC'] = None
    energy_data['predictM'] = None

    perf_dropoff = []
    energy_perf_save = []
    for idx, item in energy_data.iterrows():
        cur_app = item.appName
        cur_perf = perf_data[perf_data['kernel'] == cur_app]
        cur_pow = pow_data[pow_data['appName'] == cur_app]
        cur_perf = cur_perf.sort_values(by = ['kernel', 'coreF', 'memF']).reset_index(drop=True)
        cur_pow = cur_pow.sort_values(by = ['appName', 'coreF', 'memF']).reset_index(drop=True)

        #if cur_app == 'convolutionTexture':
        #    print cur_perf
        #    print cur_pow
        cur_perf.real = cur_perf.real / 1.0e6 / cur_perf.coreF
        cur_perf.predict = cur_perf.predict / 1.0e6 / cur_perf.coreF
        measureE = cur_perf.real * cur_pow.avg_power
        modelledE = cur_perf.predict * cur_pow.modelled_power

        defaultE_idx = cur_perf.index[(cur_perf['coreF'] == GPUCONF.CORE_FREQ) & (cur_perf['memF'] == GPUCONF.MEM_FREQ)].tolist()[0]
        defaultE = measureE[defaultE_idx]

        # get default performance
        defaultPerf = cur_perf.real[defaultE_idx]

        bestE = min(measureE)
        bestE_idx = np.argmin(measureE)
        bestC = cur_perf.loc[bestE_idx, 'coreF']
        bestM = cur_perf.loc[bestE_idx, 'memF']

        predictE_idx = np.argmin(modelledE)
        predictC = cur_perf.loc[predictE_idx, 'coreF']
        predictM = cur_perf.loc[predictE_idx, 'memF']
        #predictE = min(modelledE)
        predictE = measureE[predictE_idx]
        predictPerf = cur_perf.real[predictE_idx]

        item['defaultE'] = 1
        item['bestE'] = bestE / defaultE
        item['bestC'] = bestC
        item['bestM'] = bestM
        item['predictE'] = predictE / defaultE
        item['predictC'] = predictC 
        item['predictM'] = predictM

        #print cur_app, ":", predictPerf / defaultPerf - 1
        if (predictPerf / defaultPerf - 1) <= 0.1:
            if predictPerf <= defaultPerf:
                energy_perf_save.append(0.0)
                perf_dropoff.append(0.0)
            else:
                energy_perf_save.append(1 - predictE / defaultE)
                perf_dropoff.append(predictPerf / defaultPerf - 1)
        else:
            energy_perf_save.append(0.0)
            perf_dropoff.append(0.0)

    #print perf_dropoff
    #print energy_perf_save
    print "average dropoff:", np.mean(perf_dropoff)
    print "average energy saving within 10%% performance sacrifice: %f.[max: %f]." % (np.mean(energy_perf_save), np.max(energy_perf_save))
        
    #print energy_data
    print "average energy conservation:", 1 - np.mean(energy_data['predictE'])

if __name__ == '__main__':
    energy_best("gtx980-low-dvfs", "real-small-workload")
