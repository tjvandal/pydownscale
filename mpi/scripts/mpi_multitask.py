#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
local testing command:
mpirun -np 48 /shared/apps/sage/sage-5.12/spkg/bin/sage -python mpi_multitask.py
'''
import sys
sys.path.insert(0, "/home/vandal.t/anaconda2/lib/python2.7/site-packages")
sys.path.insert(0, "/home/vandal.t/repos/pydownscale")

from mpi4py import MPI
from scipy.stats import pearsonr, spearmanr, kendalltau
import os
import time
import numpy
from sklearn.linear_model import LinearRegression, Lasso, MultiTaskLassoCV
from sklearn.feature_selection import RFE
from sklearn.preprocessing import StandardScaler
from pydownscale.data import DownscaleData, read_nc_files
from pydownscale.downscale import DownscaleModel, ASDMultitask
from pydownscale.MSSL import pMSSL
from pydownscale import utils
import pydownscale.config as config
import argparse
import pandas
import pickle

epochs = 15
omega_epochs = 25
w_epochs_classify = 15
w_epochs = 25
size_per_job = 40
num_proc = size_per_job

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

groupworld = comm.Get_group()
num_groups = max(int(size/size_per_job), 1)
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
        rootrank = 0
        mygroupcomm = groupcomms[j]
        mygrouprank = mygroupcomm.Get_rank()
        break

if mygrouprank == 0:
    print "My group rank: %i, Processor: %s" % (mygrouprank, MPI.Get_processor_name())
    data = pickle.load(open('/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/newengland_D_12781_8835.pkl','r')) 
    total_feature_count = data.get_X().shape[1]
    y = data.get_y()[0] 

    seasons =['MAM', "DJF", 'JJA', 'SON']
    #lambda_range = numpy.logspace(numpy.log10(1e-5), numpy.log10(1e0), num=18)
    #gamma_range = numpy.logspace(numpy.log10(1e-3), numpy.log10(1e-2), num=2)
    lambda_range = [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2]
    gamma_range = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0]
    # lets split parameters
    params = numpy.array([[g, l, s] for g in gamma_range for l in lambda_range for s in seasons])
    params = numpy.array_split(params, num_groups)
    params = params[mygroupidx] 

    for i, (g, l, seas) in enumerate(params):
        pklfile = "/gss_gpfs_scratch/vandal.t/experiments/mssl_root_pkl/mssl_%s_%2.5f_%2.5f.pkl" %  ( seas, float(g), float(l))
        ncfile = "/gss_gpfs_scratch/vandal.t/experiments/mssl_root_nc/mssl_%s_%2.5f_%2.5f.nc" % ( seas, float(g), float(l))
        if os.path.exists(ncfile):
            continue
        if os.path.exists(pklfile):
            asdm = pickle.load(open(pklfile, 'r'))
        else:
            model = pMSSL(max_epochs=epochs, gamma=float(g), lambd=float(l), 
                quiet=False, omega_epochs=omega_epochs, w_epochs=w_epochs, 
                          walgo='multiprocessor',  num_proc=num_proc)
            classifier = pMSSL(max_epochs=epochs, gamma=float(g), lambd=float(l),
                quiet=False, omega_epochs=omega_epochs, w_epochs=w_epochs_classify, 
                          walgo='multiprocessor',  num_proc=num_proc, how='classify')
            #asdm = ASDMultitask(data, model, season=seas, ytransform=StandardScaler(),
            asdm = ASDMultitask(data, model, season=seas, ytransform=utils.RootTransform(0.25),
                                xtransform=StandardScaler(), max_train_year=1999,
                                conditional=classifier, cond_thres=10.)
            asdm.train()
        print "predicting"
        out = asdm.predict(test_set=True)
        out.to_netcdf(ncfile)
        pickle.dump(asdm.model, open(pklfile,  'w'))
