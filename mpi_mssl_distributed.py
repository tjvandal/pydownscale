#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
local testing command:
mpirun -np 40 /shared/apps/sage/sage-5.12/spkg/bin/sage -python mpi_mssl_distributed.py
'''
import sys
sys.path.insert(0, "/home/vandal.t/anaconda2/lib/python2.7/site-packages")

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



epochs = 15
omega_epochs = 25
w_epochs = 25
size_per_job = 48 

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
groupworld = comm.Get_group()
num_groups = max(int(size/size_per_job), 2)
jobranks = [i for i in range(size)]

# let make sure everyone has the same start time so we can save efficiently
timestr = time.strftime("%Y%m%d-%H%M%S")
timestr = comm.bcast(timestr, root=0)
if rank == 0:
    if not os.path.exists("mssl-results"):
        os.mkdir("mssl-results")

## split up job ranks 
jobranks = numpy.array_split(jobranks, num_groups)

# Build groups and communicators 
groups = [groupworld.Range_incl([(sub[0],sub[-1],1),]) for sub in jobranks]
groupcomms =[comm.Create(g) for g in groups]

for j, sub in enumerate(jobranks):
    if rank in sub:
        mygroupidx = j
        mygroup = groups[j]
        rootrank = sub[0]
        mygroupcomm = groupcomms[j]
        break

data = pickle.load(open('/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/monthly_372_7657.pkl','r')) 
#data = pickle.load(open('/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/monthly_372_780.pkl','r')) 
total_feature_count = data.get_X().shape[1]

seasons = ['DJF', 'MAM', 'JJA', 'SON']
lambda_range = numpy.logspace(numpy.log10(1e-4), numpy.log10(10.), num=6)
gamma_range = numpy.logspace(numpy.log10(1e-2), numpy.log10(10.), num=4)

seasons = ['SON']
#gamma_range=[0.0001]

# lets split parameters the good way
params = numpy.array([[g, l, s] for g in gamma_range for l in lambda_range for s in seasons])
params = numpy.array_split(params, num_groups)
params = params[mygroupidx] 

model_costs = []
model_results = []
for i, (g, l, seas) in enumerate(params):
    print "Rank: %i, Gamma: %s, Lambda: %s, Season: %s" % (rank, str(g), str(l), str(seas))	
    model = pMSSL(max_epochs=epochs, gamma=float(g), lambd=float(l), 
        quiet=False, omega_epochs=omega_epochs, w_epochs=w_epochs, walgo='mpi', mpicomm=mygroupcomm,
               mpiroot=0)
    t0 = time.time()

    #try:
    dmodel = DownscaleModel(data, model, season=seas)
    dmodel.train()
    print "Finished training at rank", rank
    mygroupcomm.Barrier()

    if rank == rootrank:
        print "getting results from the group number", mygroupidx
        res = dmodel.get_results(test_set=True)
        print "Omega Zeros: %i, W Zeros: %i" % ((dmodel.model.Omega.values == 0).sum(), (dmodel.model.W.values == 0).sum()) 
        print "Model Average Rmse: %f, Gamma: %s, Lambda: %s, Season: %s" % (numpy.mean([r['rmse'] for r in res]), g, l, seas)
        for r in res:
            r['gamma'] = g
            r['lambd'] = l
            r['omega'] = dmodel.model.Omega
            r['W'] = dmodel.model.W
            model_results.append(r)
        print "Got results from group number", mygroupidx

if rank == rootrank:
    data = pandas.DataFrame(model_results)
    data.to_pickle("mssl_downscale_results_US_%i_%s_%i.pkl" % (epochs, timestr, mygroupidx))

sys.exit()

if rank == 0:
	print "Attempting to gather Results"
	results = [r for res in model_results for r in res]
	data = pandas.DataFrame(results)
	data.to_pickle("mssl-results/mssl_US_%i_%s.pkl" % (epochs, timestr))

