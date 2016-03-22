import pickle
import numpy as np
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
from pydownscale.data import DownscaleData
from pydownscale import config
from sklearn.preprocessing import StandardScaler

import os, sys


datafile = "/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/newengland_D_12781_8835.pkl"

def load_pretraining(season):
    print "Season=", season
    data = pickle.load(open(datafile, "r"))
    season_idxs = np.where(data.observations['time.season'] == season)[0]
    times = data.observations.time[season_idxs]
    train_idxs = np.where(times['time.year'] <= config.max_train_year)
    test_idxs = np.where(times['time.year'] > config.max_train_year)
    X = data.get_X()[season_idxs][train_idxs]
    scale_x = StandardScaler().fit(X)
    X = scale_x.transform(X)
    return DenseDesignMatrix(X=X)

def load_supervised(season, which='train'):
    print "Season=", season
    data = pickle.load(open(datafile, "r"))
    season_idxs = np.where(data.observations['time.season'] == season)[0]
    times = data.observations.time[season_idxs]
    train_idxs = np.where(times['time.year'] <= config.max_train_year)
    test_idxs = np.where(times['time.year'] > config.max_train_year)
    X = data.get_X()[season_idxs]
    Y, locs = data.get_y()
    Y = Y[season_idxs]
    print "Shape of X:", X.shape
    Xtrain, Ytrain = X[train_idxs], Y[train_idxs]
    scale_x = StandardScaler().fit(Xtrain)
    if which == "train":
        Xtrain = scale_x.transform(Xtrain)
        return DenseDesignMatrix(X=Xtrain, y=Ytrain)
    elif which == 'test':
        Xtest = scale_x.transform(X[test_idxs])
        return DenseDesignMatrix(X=Xtest, y=Y[test_idxs])

if __name__ == "__main__":
    d = load_supervised("DJF")
    print d
