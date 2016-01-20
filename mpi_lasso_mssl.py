#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
local testing command:
mpirun -np 1 /shared/apps/sage/sage-5.12/spkg/bin/sage -python mpi_lasso_mssl.py
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
from pydownscale.MSSL import pMSSL
import pydownscale.config as config
import argparse
import pandas
import pickle


epochs = 100
omega_epochs = 100
w_epochs = 100

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

print "Size: %i, Rank: %i" % (size, rank)

data = pickle.load(open('/scratch/vandal.t/experiments/DownscaleData/monthly_804_420.pkl')) 
total_feature_count = data.get_X().shape[1]

#seasons = ['DJF', 'MAM', 'JJA', 'SON']
#lambda_range = numpy.logspace(numpy.log10(1e-4), numpy.log10(10.), num=10)
#gamma_range = numpy.logspace(numpy.log10(1e-2), numpy.log10(100.), num=5)

seasons = ['SON']
gamma_range=[0.01]
lambda_range = [10.0]

params = [[g, l, s] for g in gamma_range for l in lambda_range for s in seasons]
features = {}

for seas in seasons:
	if rank == 0:
		pairs = data.location_pairs('lat', 'lon')[:1]
		pairs = numpy.array_split(pairs, size)
	else:
		pairs = None

	pairs = comm.scatter(pairs, root=0)
	coefficients = []

	for p in pairs:
		model = Lasso(alpha=0.0001, normalize=True)
		dmodel = DownscaleModel(data, model, season=seas)
		dmodel.train(location={'lat': p[0], 'lon': p[1]})
		res = dmodel.get_results(test_set=True)
		print "Lasso RMSE", numpy.mean([r['rmse'] for r in res])
		coef = dmodel.model.coef_
		nonzerocoef = numpy.nonzero(coef)[0]
		coefficients.append(nonzerocoef)

	coefficients = comm.gather(coefficients, root = 0)
	
	if rank == 0:
		coefficients = [x for coef in coefficients if coef is not None for c in coef for x in c]
		columns = numpy.unique(coefficients)
		features[seas] = columns
		print "Percentage of Features Kept: %f" % (len(columns)*1./total_feature_count)
	features = comm.scatter([features]*size, root=0)

if rank == 0:
	params = numpy.array_split(numpy.asarray(params), size)
else:
	params = None

params = comm.scatter(params, root=0)
print "Rank:", rank, "Params:", params
model_costs = []
model_results = []
for g, l, seas in params:
	print "Rank: %i, Gamma: %s, Lambda: %s, Season: %s" % (rank, str(g), str(l), str(seas))	
	model = pMSSL(max_epochs=epochs, gamma=float(g), lambd=float(l), 
		quiet=False, omega_epochs=omega_epochs, w_epochs=w_epochs, walgo='admm')
	t0 = time.time()
	try:
		dmodel = DownscaleModel(data, model, season=seas, feature_columns=features[seas])
		dmodel.train()
		res = dmodel.get_results(test_set=True)
		print "Omega Zeros: %i, W Zeros: %i" % ((dmodel.model.Omega == 0).sum(), (dmodel.model.W == 0).sum()) 
		print "Model Average Rmse: %f, Gamma: %s, Lambda: %s, Season: %s" % (numpy.mean([r['rmse'] for r in res]), g, l, seas)
		for r in res:
			r['gamma'] = g
			r['lambd'] = l
			r['omega'] = dmodel.model.Omega
			r['W'] = dmodel.model.W
			model_results.append(r)

	except RuntimeWarning as err:
		pass
	except Exception as err:
		print g, l, err
		continue
		
model_results = comm.gather(model_results, root=0)

if rank == 0:
	print "Attempting to gather Results"
	results = [r for res in model_results for r in res]
	data = pandas.DataFrame(results)
	timestr = time.strftime("%Y%m%d-%H%M%S")
	data.to_pickle("mssl_downscale_results_%i_%s.pkl" % (epochs, timestr))

