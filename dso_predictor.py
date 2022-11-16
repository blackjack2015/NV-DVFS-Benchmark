import pandas as pd

from learner.models import mean_absolute_percentage_error
from learner.preprocess import piecewise_linear


def calc_dvfs_perf(Fc, delta, t0, coreFs):

    return piecewise_linear(1 / coreFs, Fc, t0, delta)


def calc_dvfs_power(p0, cg, coreFs):

    VP = 0.5

    coreVs = (coreFs - VP) ** 2 * 2 + VP
    VFs = coreVs ** 2 * coreFs

    return p0+cg*VFs


if __name__ == '__main__':

    data = pd.read_csv('data/v100-dvfs-real.csv', header=0)
    # params = pd.read_csv('data/v100-dvfs-real-params-static.csv', header=0)
    # params = pd.read_csv('data/v100-dvfs-real-params-dynamic.csv', header=0)
    params = pd.read_csv('results.csv', header=0)

    keys = ['Fc', 'delta', 't0', 'p0', 'cg']

    data = data[data.core_frequency > 700]
    data['benchmark_argNo'] = data['benchmark'] + '_' + data['argNo']
    kernels = data['benchmark_argNo'].drop_duplicates()
    kernels.sort_values(inplace=True)

    perf_mape = 0
    power_mape = 0
    for kernel in kernels:
        kdata = data[data.benchmark_argNo == kernel]
        perf = kdata["time"].values / kdata["time"].min()
        power = kdata["average_power"].values / kdata["average_power"].max()
        Fc, delta, t0, p0, cg, _ = params[params.benchmark_argNo == kernel].iloc[0].tolist()

        coreFs = kdata["core_frequency"].values * 1.0 / 1380

        perf_pred = calc_dvfs_perf(Fc, delta, t0, coreFs)
        mape = mean_absolute_percentage_error(perf, perf_pred)
        perf_mape += mape
    
        power_pred = calc_dvfs_power(p0, cg, coreFs)
        mape = mean_absolute_percentage_error(power, power_pred)
        power_mape += mape

    print("--MAPE of performance: %f%%" % (perf_mape / len(kernels)*100))
    print("--MAPE of power: %f%%" % (power_mape / len(kernels)*100))
    
