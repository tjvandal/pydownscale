import sys
import pydownscale
import pickle
import numpy
from pydownscale.data import GCMData
import pydownscale.config

gcmfile = "/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/ccsm4_historical_D.pkl"
modelfile = "/gss_gpfs_scratch/vandal.t/experiments/asd/ASD_SON.pkl"


gcm = pickle.load(open(gcmfile, "r"))
asd = pickle.load(open(modelfile, "r"))

gcm_downscaled = asd.project_gcm(gcm)

