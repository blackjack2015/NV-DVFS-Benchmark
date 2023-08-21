import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, make_scorer, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor

import xgboost as xgb
from xgboost import XGBClassifier, XGBRegressor, plot_importance

from keras.models import Sequential
from keras.layers.core import Dense, Activation
import tensorflow as tf
import keras


def mean_absolute_percentage_error(ground_truth, predictions):
    return np.mean(abs(ground_truth - predictions) / ground_truth)

def nn_fitting(X, y):

    model = Sequential()
    print("input size (num_samples, feature_dim): (%d, %d)." % (X.shape[0], X.shape[1]))
    model.add(Dense(100, input_shape=(X.shape[1],)))
    model.add(Activation('sigmoid'))
    model.add(Dense(50))
    model.add(Activation('sigmoid'))
    model.add(Dense(25))
    model.add(Activation('sigmoid'))
    model.add(Dense(5))

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=1e-3,
        decay_steps=3000
    )
    # opt = keras.optimizers.Adam(learning_rate=1e-3)
    opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanAbsoluteError())

    model.summary()

    model.fit(X, y, batch_size=16, epochs=150)   #, steps_per_epoch=3)

    return model


def xg_fitting(X, y):

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    # n_estimators = [300, 400]
    # max_depth = [3, 4]
    # learning_rate = [0.3, 0.2, 0,1, 0.05]
    # min_child_weight = [0.1, 0.5, 1, 2]
    n_estimators = [200]
    max_depth = [2]
    learning_rate = [0.1]
    min_child_weight = [0.5]
    param_grid = dict(max_depth=max_depth, n_estimators=n_estimators, learning_rate=learning_rate, min_child_weight=min_child_weight)
    
    xg_model = GridSearchCV(XGBRegressor(), cv=3, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=True)
    #xg_model = GridSearchCV(XGBRegressor(verbose=True, early_stopping_rounds=5), cv=10, param_grid=param_grid, scoring='neg_mean_squared_error', n_jobs=-1, verbose=True)
    multi_xg_model = MultiOutputRegressor(xg_model).fit(X, y)

    return multi_xg_model

def svr_fitting(X, y, kernel = 'rbf'):

    # make score function
    loss = make_scorer(mean_absolute_error, greater_is_better=False)

    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.8, 1, 1.2], 'C': [10, 100, 1000], 'epsilon': [0.1, 0.4, 0.8]},
                        {'kernel': ['poly'], 'gamma': [0.5, 1, 2], 'C': [10, 100, 1000], 'epsilon': [0.1, 0.5, 1, 2], 'degree': [2, 3, 4, 5]}]

    # tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.2], 'C': [10], 'epsilon': [0.1]},
    #                     {'kernel': ['poly'], 'gamma': [0.5], 'C': [10], 'epsilon': [0.1], 'degree': [2]}]

    kernel_idx = 0
    # initial svr model
    if kernel == 'rbf':
        kernel_idx = 0
    else:
        kernel_idx = 1

    svr_model = GridSearchCV(SVR(verbose=False, max_iter=1e6), cv=2, scoring='neg_mean_squared_error', param_grid=tuned_parameters[kernel_idx])
    #svr_model = SVR(kernel='rbf', gamma=gamma, C=C, epsilon=epsilon, verbose=True, max_iter=-1)

    # Fit regression model
    multi_svr_model = MultiOutputRegressor(svr_model).fit(X, y)

    # print svr_model.grid_scores_
    # print(svr_model.best_params_)
    # print svr_model.best_score_

    return multi_svr_model

