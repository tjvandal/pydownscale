#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
local testing command:
mpirun -np 1 /shared/apps/sage/sage-5.12/spkg/bin/sage -python mpi_linearregression.py
'''
import sys
sys.path.insert(0, "/home/vandal.t/anaconda/lib/python2.7/site-packages")

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


data = pickle.load(open('/scratch/vandal.t/experiments/DownscaleData/monthly_804_420.pkl')) 

if rank == 0:
   ## lets split up our y's
   pairs = data.location_pairs('lat', 'lon')
   print "Number of pairs:", len(pairs)
   pairs = numpy.array_split(numpy.array(pairs), size)   ## lets chunk this data up into size parts
else:
   pairs = None



pairs = comm.scatter(pairs, root=0)
print "Size:", size, "Rank:", rank, "Pairs:", pairs

results = []
models = [
          LinearRegression(normalize=True),
          LassoCV(normalize=True),
          BackwardStepwiseRegression()
        #  PCASVR(),
        #  BMA()
          ]

seasons = ['DJF', 'MAM', 'JJA', 'SON']
for p in pairs:
  for model in models[:2]:
      for season in seasons:
          try:
            #print "Rank:", rank, " Pair:", p, model.__class__.__name__, " Season:", season
            t0 = time.time()
            dmodel = DownscaleModel(data, model, season=season)
            dmodel.train(location={'lat': p[0], 'lon': p[1]})
            res = dmodel.get_results()
            for r in res:
              r['time_to_execute'] = time.time() - t0
              print "Rank: %i, Training Time: %3.2f" % (rank, r['time_to_execute'])
              results.append(r)
          except Exception as err: 
            print p, model, season, err

print "Rank: %i, Attempting to gather" % rank
newData = comm.gather(results, root=0)

if rank == 0:
    print "Rank = 0, Saving data"
    newData = [item for l in newData for item in l]   ## condense lists of lists
    data = pandas.DataFrame(newData)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    data.to_csv("linear_downscale_results_%s.csv" % timestr, index=False)
    data.to_pickle("linear_downscale_results_%s.pkl" % timestr)

