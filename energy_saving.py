import pandas as pd
import numpy as np
import sys,os
import random
from settings import *

def energy_const(gpu, version):
    
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
    elif gpu == 'v100-dvfs':
        GPUCONF = V100()

    energy_data = pd.DataFrame([])

    kernelset = perf_data['kernel'].drop_duplicates().reset_index(drop=True)
    #print kernelset

    coref_set = perf_data['coreF'].drop_duplicates().reset_index(drop=True)
    memf_set = perf_data['memF'].drop_duplicates().reset_index(drop=True)

    default_perf = perf_data[(perf_data['coreF'] == GPUCONF.CORE_FREQ) & (perf_data['memF'] == GPUCONF.MEM_FREQ)] 
    default_pow = pow_data[(pow_data['coreF'] == GPUCONF.CORE_FREQ) & (pow_data['memF'] == GPUCONF.MEM_FREQ)]

    #print default_perf
    #print default_pow

    for coref in coref_set:
        for memf in memf_set:
            energy = []
            for app in kernelset:
                dPerf = default_perf[default_perf['kernel'] == app]["real"].tolist()[0] * 1.0e-6 / GPUCONF.CORE_FREQ
                dPow = default_pow[default_pow['appName'] == app]["avg_power"].tolist()[0]

                cPerf = perf_data[(perf_data['coreF'] == coref) & (perf_data['memF'] == memf) & (perf_data['kernel'] == app)]["real"].tolist()[0] * 1.0e-6 / coref
                cPow = pow_data[(pow_data['coreF'] == coref) & (pow_data['memF'] == memf) & (pow_data['appName'] == app)]["avg_power"].tolist()[0]

                #print app, coref, memf, dPerf, dPow, cPerf, cPow
                curE = cPerf * cPow / (dPerf * dPow)
                if (cPerf < dPerf) or (cPow > dPow):
                    energy.append(1.0)
                else:
                    energy.append(curE)

            #print coref, memf, "average energy:", np.mean(energy)
            print coref, memf, "average saving:", np.mean([1.0 - item for item in energy])

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
    elif gpu == 'v100-dvfs':
        GPUCONF = V100()

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

    perf_change = []
    perf_dropoff = []
    energy_perf_save = []
    full_comp = []
    full_mem = []
    lack_wait = []
    lack_no_wait = []
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

        #print measureE
        #print modelledE

        defaultE_idx = cur_perf.index[(cur_perf['coreF'] == GPUCONF.CORE_FREQ) & (cur_perf['memF'] == GPUCONF.MEM_FREQ)].tolist()[0]
        default_type = cur_perf.loc[defaultE_idx, 'type']
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
        if predictE <= defaultE:
            item['predictE'] = predictE / defaultE
        else:
            item['predictE'] = 1.0
        item['predictC'] = predictC 
        item['predictM'] = predictM

        #print cur_app, "best core freq.:", predictC, "best mem freq.:", predictM
        #print cur_app, "best core freq.:", bestC, "best mem freq.:", bestM
        print cur_app, "default type:", default_type
        if default_type == "FULL_COMP":
            full_comp.append(1 - item['predictE'])
        elif default_type == 'FULL_MEM':
            full_mem.append(1 - item['predictE'])
        elif default_type == 'LACK_WAIT':
            lack_wait.append(1 - item['predictE'])
        elif default_type == 'LACK_NO_WAIT':
            lack_no_wait.append(1 - item['predictE'])

        #print cur_app, ":", predictPerf / defaultPerf - 1
        if (predictPerf / defaultPerf - 1) <= 0.1:
            if predictPerf <= defaultPerf:
                #energy_perf_save.append(0.0)
                #perf_dropoff.append(0.0)
                pass
            else:
                energy_perf_save.append(1 - predictE / defaultE)
                perf_dropoff.append(predictPerf / defaultPerf - 1)
        #else:
        #    energy_perf_save.append(0.0)
        #    perf_dropoff.append(0.0)

        perf_change.append(predictPerf / defaultPerf - 1)

    #print perf_dropoff
    #print energy_perf_save
    print "performance changes:", perf_change
    print "average dropoff:", np.mean(perf_dropoff)
    print "average energy saving within 10%% performance sacrifice: %f.[max: %f]." % (np.mean(energy_perf_save), np.max(energy_perf_save))
        
    print "FULL_COMP saving:", np.mean(full_comp)
    print "FULL_MEM saving:", np.mean(full_mem)
    print "LACK_WAIT saving:", np.mean(lack_wait)
    print "LACK_NO_WAIT saving:", np.mean(lack_no_wait)

    #print energy_data['predictE']
    #print energy_data
    print "average energy conservation:", np.mean(1 - energy_data['predictE']), "[max: %f]." % (1 - min(energy_data['predictE']))

if __name__ == '__main__':
    energy_best("gtx980-low-dvfs", "real-small-workload")
    energy_best("gtx1080ti-dvfs", "real")
    energy_best("p100-dvfs", "real")
    #energy_best("v100-dvfs", "real")

    energy_const("gtx980-low-dvfs", "real-small-workload")
    energy_const("gtx1080ti-dvfs", "real")
    energy_const("p100-dvfs", "real")
    #energy_const("v100-dvfs", "real")
