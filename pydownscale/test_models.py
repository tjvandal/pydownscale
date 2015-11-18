__author__ = 'tj'
import sys

from scipy.stats import pearsonr, spearmanr, kendalltau
import os
import time
import numpy
from sklearn.linear_model import LinearRegression, LassoCV, MultiTaskLassoCV
from rMSSL import pMSSL
from data import DownscaleData, read_nc_files, assert_bounds
from downscale import DownscaleModel
import config as config
from stepwise_regression import BackwardStepwiseRegression
from qrnn import QRNNCV, QRNN
from bma import BMA
import argparse
import pandas
from pcasvr import PCASVR

'''
Models to test:
Backwards stepwise regression
Lasso
Multi-task lasso
Multi-task Sparse Structure Learning
PCA-SVR
Bayesian Average modeling
Quantile Regression Neural Network

Deep Belief Network?
'''

def stepwiseregression_test(data, loc, season):
    model = BackwardStepwiseRegression()
    dmodel = DownscaleModel(data, model, season=season)
    dmodel.train(location={'lat': loc[0], 'lon': loc[1]})
    return pandas.DataFrame(dmodel.get_results())

def lasso_test(data, loc, season):
    model = LassoCV(max_iter=2000)
    dmodel = DownscaleModel(data, model, season=season)
    dmodel.train(location={'lat': loc[0], 'lon': loc[1]})
    return pandas.DataFrame(dmodel.get_results())

def mtlasso_test(data, loc, season):
    model = MultiTaskLassoCV(max_iter=2000)
    dmodel = DownscaleModel(data, model, season=season)
    dmodel.train()
    return pandas.DataFrame(dmodel.get_results())

def mssl_test(data, loc, season):
    model = pMSSL(gamma=0.1, lambd=0.1)
    dmodel = DownscaleModel(data, model, season=season)
    dmodel.train()
    return pandas.DataFrame(dmodel.get_results())

def pcasvr_test(data, loc, season):
    model = PCASVR()
    dmodel = DownscaleModel(data, model, season=season)
    dmodel.train(location={'lat': loc[0], 'lon': loc[1]})
    return pandas.DataFrame(dmodel.get_results())

def bma_test(data, loc, season):
    model = BMA()
    dmodel = DownscaleModel(data, model, season=season)
    dmodel.train(location={'lat': loc[0], 'lon': loc[1]})
    return pandas.DataFrame(dmodel.get_results())


def qrnn_test(data, loc, season):
    model = QRNNCV(tol=1e-2, hidden_nodes=[3], n_jobs=1, ntrails=3)
    #model = QRNN(tol=1e-2, ntrails=3)
    dmodel = DownscaleModel(data, model, season=season)
    dmodel.train(location={'lat': loc[0], 'lon': loc[1]})
    return pandas.DataFrame(dmodel.get_results())

if __name__ == "__main__":
    ncep_dir = config.ncep_dir
    cpc_dir = config.cpc_dir
    season = 'MAM'

    # climate model data, monthly
    ncep = read_nc_files(ncep_dir)
    ncep.load()
    ncep = assert_bounds(ncep, config.lowres_bounds)
    ncep = ncep.resample('MS', 'time', how='mean')   ## try to not resample

    # daily data to monthly
    cpc = read_nc_files(cpc_dir)
    cpc.load()
    cpc = assert_bounds(cpc, config.highres_bounds)
    monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

    data = DownscaleData(ncep, monthlycpc)
    data.normalize_monthly()
    loc = data.location_pairs('lat', 'lon')[0]
    #print "Stepwise", stepwiseregression_test(data, loc, season)
    #print "Lasso", lasso_test(data, loc, season)
    #print "MTlasso", mtlasso_test(data, loc, season)
    #print "Mssl", mssl_test(data, loc, season)
    #print "PCASVR", pcasvr_test(data, loc, season)
    #print "BMA", bma_test(data, loc, season)
    print "QRNN:", qrnn_test(data, loc, season)

