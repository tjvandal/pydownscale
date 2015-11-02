
import xray
import os, time, sys
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

    def normalize_monthly(self):
        def standardize_time(x):
            z = (x - x.mean('time')) / x.std('time')
            return z
        self.cmip = self.cmip.groupby('time.month').apply(standardize_time)
        del self.cmip['month']  ## why is the month variable created?

    def get_X(self, vars=None, timedim='time'):
        self.cmip.load()

        if vars is None:
            df = self.cmip.to_array().to_dataframe()
        else:
            df = self.cmip[vars].to_array().to_dataframe()

        levels = [var for var in df.index.names if var != timedim]
        return df.unstack(levels).values

    def get_y(self, location=None, timedim='time'):
        if location is not None:
            y = self.observations.loc[location].to_array().values.squeeze()
        else:
            y = self.observations.to_array().to_dataframe()
            levels = [var for var in y.index.names if var != timedim]
            y = y.unstack(levels)
            location = y.columns.to_series()
            y = y.values

        return y, location

    def location_pairs(self, dim1, dim2):
        Y = self.observations.to_array()
        if (dim1 not in Y.dims) or (dim2 not in Y.dims):
            raise IndexError("dim1=%s and dim2=%s are not in observations." % (dim1, dim2))
        if len(Y.dims) != 4:
            raise ValueError("There should be 4 dimensions with only 1 variable.")

        t0 = Y['time'].values[0]
        t0array = self.observations.loc[dict(time=t0)].to_array()

        # check which dimension axes
        dims = t0array.dims
        dim1_axis = [j for j in range(len(dims)) if dims[j] == dim1][0]
        dim2_axis = [j for j in range(len(dims)) if dims[j] == dim2][0]
        t0values = t0array.values
        if dim1_axis > dim2_axis:
            t0values = numpy.swapaxes(t0values, dim1_axis, dim2_axis)
        t0values = t0values.squeeze()

        pairs = [[d1, d2] for i1, d1 in enumerate(Y.coords[dim1].values) for i2, d2 in enumerate(Y.coords[dim2].values) if t0values[i1, i2] != -999.]
        return pairs


def read_nc_files(dir):
    def rmheight(d):
        #del d["height"]
        return d

    files = [os.path.join(dir, f) for f in os.listdir(dir) if ".nc" == f[-3:]]
    if len(files) > 1:
        data = xray.open_mfdataset(files, preprocess=rmheight)
    elif len(files) == 1:
        data = xray.open_dataset(files[0])
    else:
        raise IOError("There are no .nc files in that directory.")
    return data

def assert_bounds(data, bounds):
    '''
    :param data: Should be an xray dataset
    :param bounds: dictionary of bounds {'lat': [l1 ,l2], 'lon': [l3, l4]}
    :return: xray dataset bounded by the bounds
    '''
    loc = {}
    for b in bounds:
        vals = data[b].values
        loc[b] = vals[(bounds[b][0] < vals) & (bounds[b][1] > vals)]
    return data.loc[loc]

if __name__ == "__main__":
    cmip5_dir = "/scratch/vandal.t/cmip5/historical/atm/mon/MIROC-ESM/"
    cpc_dir = "/Users/tj/data/usa_cpc_nc/merged"

    # climate model data, monthly
    cmip5 = read_nc_files(cmip5_dir)
    cmip5.load()
    cmip5 = assert_bounds(cmip5, {'lat': [20, 35], 'lon': [100, 150]})

    cmip5.time = pandas.to_datetime(cmip5.time.values)
    cmip5 = cmip5.resample('MS', 'time', how='mean')   ## try to not resample

    # daily data to monthly
    cpc = read_nc_files(cpc_dir)
    cpc.load()
    cpc.time = pandas.to_datetime(cpc.time.values)
    cpc = assert_bounds(cpc, {'lat': [20, 35], 'lon': [-120, -100]})

    print "resampling cpc"
    monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

    d = DownscaleData(cmip5, monthlycpc)
    print "Normalizing"
    d.normalize_monthly()
    X = d.get_X(vars=['uas', 'vas', 'tasmax', 'tasmin', 'hurs'])
    print d.location_pairs('lat', 'lon')

    lt = d.cmip['lat'].values[0]
    ln = d.cmip['lon'].values[0]
    prcp = d.cmip.loc[{'lat':  lt, 'lon': ln}]

    from matplotlib import pyplot
    import seaborn

    pyplot.hist(prcp['tas'].values)
    pyplot.show()