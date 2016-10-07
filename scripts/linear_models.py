import sys
import numpy
from pydownscale.data import DownscaleData
from pydownscale import config
import xarray
from pydownscale.utils import LogTransform, BoxcoxTransform
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoLarsCV,LinearRegression
from pydownscale.downscale import ASD
import pickle
from pydownscale.stepwise_regression import BackwardStepwiseRegression
from pydownscale.bma import BMA

datafile = '/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/newengland_D_12783_8835.pkl'
data = pickle.load(open(datafile, 'r'))
seasons = ['SON', 'DJF', 'MAM', 'JJA']
for seas in seasons:
    models = {"lasso": ASD(data, nearest_neighbor=False, model=LassoLarsCV(), season=seas, 
                       xtransform=BoxcoxTransform(), ytransform=LogTransform(), num_proc=16),
          "ASD": ASD(data, nearest_neighbor=True, model=LinearRegression(), season=seas, 
                     xtransform=StandardScaler(), ytransform=StandardScaler(), num_proc=48,
                    training_size=0.10), 
          "BMA": ASD(data, nearest_neighbor=True, model=BMA(),season=seas, 
                     xtransform=BoxcoxTransform(), ytransform=LogTransform(), num_proc=48)}
    for key, model in models.iteritems():
        if key != 'ASD':
            continue
        print "training model %s for season %s" % (key, seas)
        model.train()
        predicted = model.predict(test_set=False)
        predicted.to_netcdf("/gss_gpfs_scratch/vandal.t/experiments/asd_nc/%s_D_%s.nc" % (key, seas))
        pickle.dump(model, open("/gss_gpfs_scratch/vandal.t/experiments/asd/%s_%s.pkl" % (key, seas), "w"))


