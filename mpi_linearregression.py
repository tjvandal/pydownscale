#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
local testing command:
mpirun -np 2 python mpi_linearregression.py --ncep_dir /Users/tj/data/ncep/access1-3/ --cpc_dir /Users/tj/data/usa_cpc_nc/merged/
'''
import sys
sys.path.insert(0, "/home/vandal.t/anaconda/lib/python2.7/site-packages")

from mpi4py import MPI
from scipy.stats import pearsonr, spearmanr, kendalltau
import os
import time
import numpy
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import RFE
from pydownscale.data import DownscaleData, read_nc_files
from pydownscale.downscale import DownscaleModel
import pydownscale.config as config
import argparse
import pandas

ncep_dir = config.ncep_dir
cpc_dir = config.cpc_dir

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

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

if rank == 0:
   ## lets split up our y's
   pairs = data.location_pairs('lat', 'lon')
   pairs = numpy.split(pairs, size)   ## lets chunk this data up into size parts
else:
   pairs = None

pairs = comm.scatter(pairs, root=0)
results = []
models = [LassoCV(alphas=[1, 10, 100])]
seasons = ['DJF', 'MAM', 'JJA', 'SON']
for p in pairs:
  for model in models:
      for season in seasons:
          t0 = time.time()
          dmodel = DownscaleModel(data, model, season=season)
          dmodel.train(location={'lat': p[0], 'lon': p[1]})
          res = dmodel.get_results()
          res['time_to_execute'] = time.time() - t0
          results.append(res)

newData = comm.gather(results, root=0)

if rank == 0:
   newData = [item for l in newData for item in l]   ## condense lists of lists
   data = pandas.DataFrame(newData)
   data.to_csv("results.csv", index=False)
