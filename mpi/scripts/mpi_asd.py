#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
local testing command:
mpirun -np 48 /shared/apps/sage/sage-5.12/spkg/bin/sage -python mpi_asd.py
'''
import sys, os
sys.path.insert(0, "/home/vandal.t/anaconda2/lib/python2.7/site-packages")
pydownscale_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, pydownscale_dir)

from mpi4py import MPI
import numpy
import pickle
import xarray

from pydownscale.data import DownscaleData
from pydownscale import config
from pydownscale import utils
from pydownscale.utils import LogTransform, BoxcoxTransform
from pydownscale.downscale import ASD
from pydownscale.stepwise_regression import BackwardStepwiseRegression
from pydownscale.bma import BMA

from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsCV,LinearRegression, LogisticRegression
from sklearn import linear_model

def main():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    print "Size: %i, Rank: %i" % (size, rank)

    size_per_job = 40
    groupworld = comm.Get_group()
    num_groups = max(int(size/size_per_job), 1)
    jobranks = [i for i in range(size)]
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
        datafile = '/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/newengland_D_12781_8835.pkl'
        data = pickle.load(open(datafile, 'r'))
        jobs = []
        seasons = ['SON', 'DJF', 'MAM', 'JJA']
        for seas in seasons:
            jobs += [
                 #["ELNET", seas,  ASD(data, nearest_neighbor=False, 
                 #         model=linear_model.ElasticNetCV(l1_ratio=[0.1,0.25,0.5,0.75,0.9,1.], n_alphas=10),
                 #          season=seas, xtransform=StandardScaler(), cond_thres=10.,
                 #                     conditional=LogisticRegression("l1", C=0.1),
                 #         ytransform=utils.RootTransform(0.24), num_proc=size_per_job)], 
                #["PCAOLS", seas,  ASD(data, nearest_neighbor=False, model=LinearRegression(),season=seas, 
                #             xtransform=PCA(n_components=0.98), ytransform=utils.RootTransform(0.25),
                #                      num_proc=size_per_job, conditional=LogisticRegression("l2", C=10.),
                #                      cond_thres=10.)],
                 # ["PCASVRRoot", seas,  ASD(data, nearest_neighbor=False, model=svm.LinearSVR(),season=seas, 
                 #            xtransform=PCA(n_components=0.98), ytransform=utils.RootTransform(0.25),
                 #            num_proc=size_per_job, conditional=svm.SVC(kernel='linear'),
                 #                           cond_thres=10.)],
                  ["PCASVRLogistic", seas,  ASD(data, nearest_neighbor=False, model=svm.LinearSVR(),season=seas, 
                             xtransform=PCA(n_components=0.98), ytransform=utils.RootTransform(0.25),
                             num_proc=size_per_job, conditional=LogisticRegression("l2", C=10.), cond_thres=10.)]
                ]
        jobsplit = numpy.array_split(jobs, num_groups)
        myjobs = jobsplit[mygroupidx]
        for key, seas, model in myjobs:
            try:
                print "training model %s for season %s" % (key, seas)
                model.train()
                predicted = model.predict(test_set=True)
                pickle.dump(model, open("/gss_gpfs_scratch/vandal.t/experiments/asd_pkl/%s_%s.pkl" % (key, seas), "w"))
                predicted.to_netcdf("/gss_gpfs_scratch/vandal.t/experiments/asd_nc/%s_D_%s.nc" % (key, seas))
                del model
            except Exception as err:
                print key, err

if __name__ == "__main__":
    main()
