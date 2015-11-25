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
from sklearn.linear_model import LinearRegression, LassoCV, MultiTaskLassoCV
from sklearn.feature_selection import RFE
from pydownscale.data import DownscaleData, read_nc_files
from pydownscale.downscale import DownscaleModel
from pydownscale.rMSSL import pMSSL
import pydownscale.config as config
import argparse
import pandas
import pickle


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = pickle.load(open('/scratch/vandal.t/experiments/DownscaleData/monthly_804_3150.pkl')) 
pairs = data.location_pairs('lat', 'lon')

if rank == 0:
	seasons = ['DJF', 'MAM', 'JJA', 'SON']
else:
	seasons = None
	
seas = comm.scatter(seasons, root=0)
model = pMSSL(quite=True)

t0 = time.time()
dmodel = DownscaleModel(data, model, season=seas)
dmodel.train()
res = dmodel.get_results()
#r['time_to_execute'] = time.time() - t0

newData = comm.gather(res, root=0)

if rank == 0:
   newData = [item for l in newData for item in l]   ## condense lists of lists
   data = pandas.DataFrame(newData)
   timestr = time.strftime("%Y%m%d-%H%M%S")
   data.to_csv("mssl_downscale_results_%s.csv" % timestr, index=False)
   data.to_pickle("mssl_downscale_results_%s.pkl" % timestr)
