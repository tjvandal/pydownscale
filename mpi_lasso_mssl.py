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
from pydownscale.rMSSL import pMSSL
import pydownscale.config as config
import argparse
import pandas
import pickle


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

data = pickle.load(open('/scratch/vandal.t/experiments/DownscaleData/monthly_804_3150.pkl')) 
total_feature_count = data.get_X().shape[1]

seasons = ['DJF', 'MAM', 'JJA', 'SON'] #[:1]
lambda_range = numpy.logspace(numpy.log10(1e-4), numpy.log10(2.), num=10)
gamma_range = numpy.logspace(numpy.log10(1e-4), numpy.log10(2.), num=10)
params = [[g, l, s] for g in gamma_range for l in lambda_range for s in seasons]
print "Number of Parameter Sets:", len(params)
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
		model = Lasso(alpha=0.05, normalize=True)
		dmodel = DownscaleModel(data, model, season=seas)
		dmodel.train(location={'lat': p[0], 'lon': p[1]})
		res = dmodel.get_results(test_set=False)
		print "Lasso Pearson", numpy.mean([r['spearman'] for r in res])
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
	params = numpy.array_split(params, size)
else:
	params = None

params = comm.scatter(params, root=0)
model_costs = []
model_results = []
for g, l, seas in params:
	model = pMSSL(max_epochs=30, gamma=float(g), lambd=float(l), quite=False)
	#model = pMSSL(max_epochs=100, gamma=0., lambd=0., quite=False, wadmm=True)
	t0 = time.time()
	try:
		dmodel = DownscaleModel(data, model, season=seas, feature_columns=features[seas])
		dmodel.train()
		res = dmodel.get_results(test_set=True)
		print "Model Average Pearson", numpy.mean([r['pearson'] for r in res])
	except RuntimeWarning as err:
		pass
	except Exception as err:
		print g, l, err
		continue

	# cost = dmodel.model.cost(dmodel.X_train, dmodel.y_train, dmodel.model.W, dmodel.model.Omega)
	# model_costs.append([seas, cost, dmodel])

	res = dmodel.get_results()
	for r in res:
		r['gamma'] = g
		r['lambd'] = l
		r['omega'] = dmodel.model.Omega
		r['W'] = dmodel.model.W
		model_results.append(r)

model_results = comm.gather(model_results, root=0)

if rank == 0:
	results = [r for res in model_results for r in res]
	data = pandas.DataFrame(results)
	timestr = time.strftime("%Y%m%d-%H%M%S")
	data.to_csv("mssl_downscale_results_%s.csv" % timestr, index=False)
	data.to_pickle("mssl_downscale_results_%s.pkl" % timestr)

