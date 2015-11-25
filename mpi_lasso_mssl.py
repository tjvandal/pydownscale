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
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLassoCV
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

data = pickle.load(open('/scratch/vandal.t/experiments/DownscaleData/monthly_804_120.pkl')) 
total_feature_count = data.get_X().shape[1]
seasons = ['DJF', 'MAM', 'JJA', 'SON'][:2]
features = {}

for seas in seasons:
	if rank == 0:
		pairs = data.location_pairs('lat', 'lon')
		pairs = numpy.array_split(pairs, size)
	else:
		pairs = None

	pairs = comm.scatter(pairs, root=0)
	coefficients = []

	for p in pairs:
		model = Lasso(alpha=0.001, normalize=True)
		dmodel = DownscaleModel(data, model, season=seas)
		dmodel.train(location={'lat': p[0], 'lon': p[1]})
		coef = dmodel.model.coef_
		nonzerocoef = numpy.nonzero(coef)[0]
		coefficients.append(nonzerocoef)

	coefficients = comm.gather(coefficients, root = 0)
	#if rank == 0:
	coefficients = [x for coef in coefficients for c in coef for x in c]
	columns = numpy.unique(coefficients)
	#print columns
	features[seas] = columns
	print "Percentage of Features Kept: %f" % (len(columns)*1./total_feature_count)

if rank == 0:
	seasons = numpy.array_split(seasons, size)
else:
	seasons = None

seasons = comm.scatter(seasons, root=0)
results = []
for seas in seasons:
	model = pMSSL(quite=True)
	t0 = time.time()
	print features[seas]
	dmodel = DownscaleModel(data, model, season=seas, feature_columns=features[seas])
	dmodel.train()
	res = dmodel.get_results()
	results.append(res)

newData = comm.gather(results, root=0)

if rank == 0:
   newData = [item for l in newData for item in l]   ## condense lists of lists
   data = pandas.DataFrame(newData)
   timestr = time.strftime("%Y%m%d-%H%M%S")
   data.to_csv("mssl_downscale_results_%s.csv" % timestr, index=False)
   data.to_pickle("mssl_downscale_results_%s.pkl" % timestr)
