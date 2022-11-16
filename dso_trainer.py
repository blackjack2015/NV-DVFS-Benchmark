import pandas as pd

from learner.preprocess import normalize
from learner.preprocess import get_ptx_types, get_inst_types, get_mem_types
from learner.models import nn_fitting, mean_absolute_percentage_error


if __name__ == '__main__':

    train_data = pd.read_csv('data/v100-dvfs-microbenchmark.csv', header=0)
    test_data = pd.read_csv('data/v100-dvfs-real.csv', header=0)

    train_data = normalize(train_data)
    test_data = normalize(test_data)

    train_data.to_csv('data/train.csv', index=False)
    test_data.to_csv('data/test.csv', index=False)

    params = ['Fc', 'delta', 't0', 'p0', 'cg']
    keys = []
    dcgm_metrics = ['DRAMA', 'FP16A', 'FP32A', 'FP64A', 'SMACT', 'SMOCC', 'TENSO']
    keys.extend(dcgm_metrics)
    keys.extend(get_ptx_types())
    keys.extend(get_inst_types())
    keys.extend(get_mem_types())

    # Fc, cg, delta, p0, t0
    train_X = train_data[keys]
    train_y = train_data['t0']
    train_y = train_data[['Fc', 'delta', 't0', 'p0', 'cg']]
    test_X = test_data[keys]
    test_y = test_data['t0']
    test_y = test_data[['Fc', 'delta', 't0', 'p0', 'cg']]

    print(train_X)
    print(train_y)

    model = nn_fitting(train_X, train_y)

    train_y_pred = model.predict(train_X)
    train_mae = mean_absolute_percentage_error(train_y, train_y_pred)
    print(train_mae)

    test_y_pred = model.predict(test_X)
    test_mae = mean_absolute_percentage_error(test_y, test_y_pred)
    print(test_mae)

    df_params = pd.DataFrame(test_y_pred, columns=params)
    df_params["benchmark_argNo"] = test_data["benchmark_argNo"]

    print(df_params.head())
    df_params.to_csv("results.csv", index=False)
    
