import os
from scipy import optimize
import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
import pwlf


ASSEMBLY_INFO_ROOT = 'ptx_tools/assembly_info/'


def LR(X, y, alpha = 5e-4): 
    
    #lin = Lasso(alpha=0.0001,precompute=True,max_iter=1000, positive=True, random_state=9999, selection='random', fit_intercept=False)
    lin = Lasso(alpha=alpha, positive=True, fit_intercept=False)
    lin.fit(X,y)
    #print(lin.coef_, lin.intercept_)
    w = lin.coef_
    #w = np.array([lin.intercept_, lin.coef_[0], lin.coef_[1]])
    #print(w)

    #inv_item = np.linalg.inv((np.dot(X.T, X)) + alpha * np.identity(X.shape[1]))
    #w = np.dot((inv_item @ X.T), y)

    return w


def get_ptx_types():

    ptx_type_file = os.path.join(ASSEMBLY_INFO_ROOT, 'ptx_instruction_types.txt')
    with open(ptx_type_file) as f:
        dtypes = f.readlines()

    dtypes = [item.strip() for item in dtypes]

    return dtypes


def get_inst_types():

    inst_type_file = os.path.join(ASSEMBLY_INFO_ROOT, 'ptx_isa.txt')
    with open(inst_type_file) as f:
        inst_types = f.readlines()

    inst_types = [item.strip() for item in inst_types]

    return inst_types


def get_mem_types():

    mem_type_file = os.path.join(ASSEMBLY_INFO_ROOT, 'ptx_state_spaces.txt')
    with open(mem_type_file) as f:
        mem_types = f.readlines()

    mem_types = [item.strip() for item in mem_types]

    return mem_types


def piecewise_linear(x, x0, y0, k1):
    return np.piecewise(x, [x < x0], [lambda x: y0, lambda x: y0+(x-x0)*k1])


def fit_piecewise_dvfs_perf_model(data):

        # return np.piecewise(x, [x < x0], [lambda x: y0, lambda x: x0+k1*x])

    CORE_BASE = 1380.0
    MEM_BASE = 877.0

    coreFs = data["core_frequency"].values * 1.0 / CORE_BASE
    
    # fitting performance
    x = 1 / coreFs
    y = data["time"].values / data["time"].min()

    popt, pconv = optimize.curve_fit(piecewise_linear, x, y)
    x0, y0, k1 = popt
    rec_y = piecewise_linear(x, x0, y0, k1)

    return x0, y0, k1, np.mean((y - rec_y) ** 2), np.mean(np.abs(y - rec_y)/y)


def fit_dvfs_power_model(data):

    CORE_BASE = 1380.0
    MEM_BASE = 877.0
    VP = 0.5

    coreFs = data["core_frequency"].values * 1.0 / CORE_BASE
    coreVs = (coreFs - VP) ** 2 * 2 + VP
    # memFs = tmp_df["memory_frequency"].values * 1.0 / MEM_BASE
    # fitting power 
    VFs = coreVs ** 2 * coreFs
    # X = np.stack((np.ones(memFs.shape[0]), memFs, VFs)).T
    X = np.stack((np.ones(VFs.shape[0]), VFs)).T
    y = data["average_power"].values / data[data.core_frequency == 1380].average_power.item()
    pw = LR(X, y)
    p0 = pw[0]
    # gamma = pw[1]
    cg = pw[1]
    # rec_y = p0 + gamma*memFs + cg*VFs
    rec_y = p0 + cg*VFs
    #rec_y = pw.predict(X)

    return p0, cg, np.mean((y - rec_y) ** 2), np.mean(np.abs(y - rec_y)/y)


def fit_dvfs_model(data):

    model_results = pd.DataFrame()

    data = data[data.core_frequency > 700]
    kernels = data['benchmark_argNo'].drop_duplicates()
    kernels.sort_values(inplace=True)
    
    time_err = 0
    power_err = 0
    time_mape = 0
    power_mape = 0
    p_basic = []
    gamma = []
    cg = []
    print("Benchmarks:")
    for kernel in kernels:
    
        tmp_df = data[data.benchmark_argNo == kernel]

        Fc, t0, delta, err, mape = fit_piecewise_dvfs_perf_model(tmp_df)

        time_err += err
        time_mape += mape

        p0, cg, err, mape = fit_dvfs_power_model(tmp_df)
        power_err += err
        power_mape += mape

        one_data = {}
        one_data['benchmark_argNo'] = kernel
        params = {"p0":p0, "cg":cg, "t0":t0, "Fc":Fc, "delta":delta}
        one_data.update(params)
        model_results = model_results.append(one_data, ignore_index=True)

    print("Error Measurement:")
    print("--RMSE of performance:", time_err / len(kernels))
    print("--RMSE of power:", power_err / len(kernels))
    print("--MAPE of performance: %f%%" % (time_mape / len(kernels)*100))
    print("--MAPE of power: %f%%" % (power_mape / len(kernels)*100))

    return model_results


def normalize(data):
    """
       data is the raw metrics collectd by DCGM and PTX.
    """

    dtypes = get_ptx_types()
    inst_types = get_inst_types()
    mem_types = get_mem_types()

    # normalize ptx info within each domain
    data[dtypes] = data[dtypes].div(data[dtypes].sum(axis=1), axis=0)
    data[inst_types] = data[inst_types].div(data[inst_types].sum(axis=1), axis=0)
    data[mem_types] = data[mem_types].div(data[mem_types].sum(axis=1), axis=0)

    data['benchmark_argNo'] = data['benchmark'] + '_' + data['argNo']
    kernel_params = fit_dvfs_model(data[['benchmark_argNo', 'core_frequency', 'memory_frequency', 'time', 'average_power']])

    print(kernel_params)

    data = data[data.core_frequency == 1380]

    data = data.merge(kernel_params, on='benchmark_argNo')
    return data


if __name__ == '__main__':

    data = pd.read_csv('data/v100-dvfs-real.csv', header=0)
    data = normalize(data)
    data.to_csv('data/v100-dvfs-real-preprocess.csv', index=False)
    data = pd.read_csv('data/v100-dvfs-microbenchmark.csv', header=0)
    data = normalize(data)
    data.to_csv('data/v100-dvfs-microbenchmark-preprocess.csv', index=False)
