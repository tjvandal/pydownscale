
import xray
import os, time
import numpy
import pandas
from scipy.misc import factorial

class DownscaleData:
    def __init__(self, cmip, observations):
        self.cmip = cmip
        self.observations = observations
        self._checkindices()
        self._matchdates()

    # can we make this dynamic in the future?
    def _checkindices(self):
        if not 'time' in self.cmip.dims:
            raise IndexError("time should be a dimension of cmip")

        if not 'time' in self.observations.dims:
            raise IndexError("time should be a dimension of observations")

    # Line up datasets so that the times match.  Exclude observations outside of the timeframe.
    def _matchdates(self):
        mintime = max(self.cmip.time.min(), self.observations.time.min())
        maxtime = min(self.cmip.time.max(), self.observations.time.max())

        cmiptimes = (self.cmip.time >= mintime) & (self.cmip.time <= maxtime)
        obstimes = (self.observations.time >= mintime) & (self.observations.time <= maxtime)

        self.cmip = self.cmip.loc[dict(time=self.cmip.time[cmiptimes])]
        self.observations = self.observations.loc[dict(time=self.observations.time[obstimes])]

        if (len(self.cmip.time) != len(self.observations.time)) or \
            sum(self.cmip.time == self.observations.time) != len(self.cmip.time):
            raise IndexError("times do not match.  add functionality if this is not an error")

    def get_covariate_indices(self):
        return self.cmip.flatten()


    def get_X(self):
        self.cmip.load()
        X = self.cmip.to_array().values
        dims = self.cmip.to_array().dims
        times_axis = [j for j in range(len(dims)) if dims[j] == 'time'][0]
        X = numpy.swapaxes(X, times_axis, 0)
        ndim = 1
        for j in X.shape[1:]:
            ndim *= j
        X = X.reshape((X.shape[0], ndim))
        return X

def read_nc_files(dir):
    files = [os.path.join(dir, f) for f in os.listdir(dir) if ".nc" == f[-3:]]
    if len(files) > 1:
        data = xray.open_mfdataset(files)
    elif len(files) == 1:
        data = xray.open_dataset(files[0])
    else:
        raise IOError("There are no .nc files in that directory.")
    return data

if __name__ == "__main__":
    cmip5_dir = "/Users/tj/data/cmip5/access1-0/"
    cpc_dir = "/Users/tj/data/usa_cpc_nc/merged"

    # climate model data, monthly
    cmip5 = read_nc_files(cmip5_dir)
    #cmip5.load()
    cmip5.time = pandas.to_datetime(cmip5.  time.values)
    cmip5 = cmip5.resample('MS', 'time', how='mean')   ## try to not resample

    # daily data to monthly
    cpc = read_nc_files(cpc_dir)
    cpc.time = pandas.to_datetime(cpc.time.values)
    #cpc.load()
    print "resampling cpc"
    monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

    d = DownscaleData(cmip5, monthlycpc)
    print "getting covariates"
    X = d.get_X()
