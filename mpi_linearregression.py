#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
local testing command:
mpirun -np 1 /shared/apps/sage/sage-5.12/spkg/bin/sage -python mpi_linearregression.py
'''
import sys
sys.path.insert(0, "/home/vandal.t/anaconda2/lib/python2.7/site-packages")

from mpi4py import MPI
from scipy.stats import pearsonr, spearmanr, kendalltau
import os
import time
import numpy
from sklearn.linear_model import LinearRegression, LassoCV, MultiTaskLassoCV
from sklearn.feature_selection import RFE
from pydownscale.data import DownscaleData, read_nc_files
from pydownscale.downscale import DownscaleModel
from pydownscale.stepwise_regression import BackwardStepwiseRegression
from pydownscale.pcasvr import PCASVR
from pydownscale.bma import BMA
import pydownscale.config as config
import argparse
import pandas
import pickle


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()



data = pickle.load(open('/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/monthly_324_7657.pkl','r')) 
pairs = data.location_pairs('lat', 'lon')
seasons = ['DJF', 'MAM', 'JJA', 'SON']
models = [
#          LinearRegression(normalize=True),
#          LassoCV(normalize=True),
          BackwardStepwiseRegression(),
        #  PCASVR(),
        #  BMA()
          ]

params = [[model, season, pair] for model in models for season in seasons for pair in pairs]

if rank == 0:
   ## lets split up our y's 
   print "Number of pairs:", len(pairs)
   params = numpy.array_split(numpy.array(params), size)   ## lets chunk this data up into size parts
else:
   params = None



params = comm.scatter(params, root=0)
print "Size:", size, "Rank:", rank, "Params:", params

results = []

for model,season, p in params:
    try:
        t0 = time.time()
        dmodel = DownscaleModel(data, model, season=season)
        dmodel.train(location={'lat': p[0], 'lon': p[1]})
        #pickle.dump(dmodel, open("%s_%2.2d_%2.2f_%s" % (model.__class__.__name__, p[0],
        #                                                p[1],season),'w')) 
        res = dmodel.get_results()
        for r in res:
          r['time_to_execute'] = time.time() - t0
          print "Rank: %i, Training Time: %3.2f" % (rank, r['time_to_execute'])
          results.append(r)
    except Exception as err:
        print err
        print "Error:", p, model.__class__.__name__, season, err

print "Rank: %i, Attempting to gather" % rank
newData = comm.gather(results, root=0)

if rank == 0:
    print "Rank = 0, Saving data"
    newData = [item for l in newData for item in l]   ## condense lists of lists
    data = pandas.DataFrame(newData)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #data.to_csv("linear_downscale_results_day_%s.csv" % timestr, index=False)
    data.to_pickle("linear_downscale_results_MS_%s.pkl" % timestr)

