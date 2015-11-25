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
from sklearn.linear_model import LinearRegression, LassoCV, MultiTaskLasso
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


seasons = ['DJF', 'MAM', 'JJA', 'SON']
alphas = numpy.logspace(numpy.log10(0.001), numpy.log10(10),
                       num=100)[::-1]
results = []

for seas in seasons:
	if rank == 0:
		alphas = numpy.array_split(alphas, size)
	else:
		alphas = None

	alphas = comm.scatter(alphas, root=0)

	cv_results = []

	for alpha in alphas:
		model = MultiTaskLasso(alpha=alpha, max_iter=3000, normalize=True)
		t0 = time.time()
		dmodel = DownscaleModel(data, model, season=seas)
		dmodel.train()
		res = dmodel.get_results()
		rmse = numpy.mean([r['rmse'] for r in res])
		cv_results.append([alpha, rmse])

	cvdata = comm.gather(cv_results, root=0)

	if rank == 0:
		cvdata = [item for l in cvdata for item in l]   ## condense lists of lists
		cvdata = numpy.array(cvdata)
		idx = numpy.argmin(cvdata[:,1])
		alpha = cvdata[idx,0]
		model = MultiTaskLasso(alpha=alpha, max_iter=3000, normalize=True)
		t0 = time.time()
		dmodel = DownscaleModel(data, model, season=seas)
		dmodel.train()
		res = dmodel.get_results()
		for r in res:
			r['time_to_execute'] = time.time() - t0
			results.append(r)

if rank == 0:
	print results
	#results = [item for l in results for item in l]   ## condense lists of lists
	
	print results
	data = pandas.DataFrame(results)
	timestr = time.strftime("%Y%m%d-%H%M%S")
	data.to_csv("mtl_downscale_results_%s.csv" % timestr, index=False)
	data.to_pickle("mtl_downscale_results_%s.pkl" % timestr)
