
import xray
import os, time, sys
import numpy
import pandas
from scipy.misc import factorial

'''
Lets let ncep be a list of xray datasets where the time axis matches but may have different levels
normalization is no needed at this step, atleast until training
'''

class DownscaleData:
    def __init__(self, ncep, observations):
        if isinstance(ncep, xray.Dataset):
            ncep = [ncep]

        self.ncep = ncep
        self.observations = observations
        self._check_ncep()
        self._checkindices()
        self._matchdates()

    def _check_ncep(self):
        for n in self.ncep:
            if (not isinstance(n, xray.Dataset)) and (not isinstance(n, xray.DataArray)):
                raise TypeError("Ncep should be a list of xray datasets or dataarrays.")

    # can we make this dynamic in the future?
    def _checkindices(self):
        times = []
        for d in self.ncep:
            if not 'time' in d.dims:
                raise IndexError("time should be a dimension of cmip")
            times.append(d.time.values)

        times = numpy.array(times)
        if numpy.sum(times[0] - times) != 0:
            raise IndexError("All times in lowres data should be identical.")

        if not 'time' in self.observations.dims:
            raise IndexError("time should be a dimension of observations")

    # Line up datasets so that the times match.  Exclude observations outside of the timeframe.
    def _matchdates(self):
        mintime = max(self.ncep[0].time.min(), self.observations.time.min())
        maxtime = min(self.ncep[0].time.max(), self.observations.time.max())

        cmiptimes = (self.ncep[0].time >= mintime) & (self.ncep[0].time <= maxtime)
        obstimes = (self.observations.time >= mintime) & (self.observations.time <= maxtime)

        for j in range(len(self.ncep)):
            self.ncep[j] = self.ncep[j].loc[dict(time=self.ncep[j].time[cmiptimes])]

        self.observations = self.observations.loc[dict(time=self.observations.time[obstimes])]

        if (len(self.ncep[j].time) != len(self.observations.time)) or \
            sum(self.ncep[j].time == self.observations.time) != len(self.ncep[j].time):
            raise IndexError("times do not match.  add functionality if this is not an error")

    def get_covariate_indices(self):
        return self.ncep.flatten()

    def normalize_monthly(self):
        def standardize_time(x):
            z = (x - x.mean('time')) / x.std('time')
            return z
        self.ncep = self.ncep.groupby('time.month').apply(standardize_time)
        del self.ncep['month']  ## why is the month variable created?

    def get_X(self, timedim='time'):
        x = []
        for d in self.ncep:
            d.load()
            df = d.to_array().to_dataframe()
            levels = [var for var in df.index.names if var != timedim]
            x.append(df.unstack(levels).values)

        x = numpy.column_stack(x)
        print x.shape
        return x


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

        if isinstance(t0values, float):
            t0values = numpy.array(t0values)
            t0values = t0values[:, numpy.newaxis]
        elif len(Y.coords[dim1].values) == 1:
            t0values = t0values[numpy.newaxis, :]
        elif len(Y.coords[dim2].values) == 1:
            t0values = t0values[:, numpy.newaxis]

        pairs = [[d1, d2] for i1, d1 in enumerate(Y.coords[dim1].values) for i2, d2 in enumerate(Y.coords[dim2].values) if t0values[i1, i2] != -999.]
        return pairs

def get_ncep_file_paths(basedir):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(basedir):
        for f in filenames:
            if '.nc' in f:
                files.append(os.path.join(dirpath, f))

    return sorted(files)


def read_nc_files(dir):
    def rmheight(d):
        #del d["height"]
        return d

    files = get_ncep_file_paths(dir)
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

def test_data():
    p1 = "/scratch/vandal.t/ncep/daily/downscale-data/air/"
    p2 = "/scratch/vandal.t/ncep/daily/downscale-data/pr_wtr/"
    files1 = get_ncep_file_paths(p1)[2:3]
    files2 = get_ncep_file_paths(p2)[2:3]

    d1 = xray.open_mfdataset(files1)
    d2 = xray.open_mfdataset(files2)
    lowres = []
    for d in [d1, d2]:
        d.load()
        d = assert_bounds(d, config.lowres_bounds)
        if 'level' in d.dims:
            d = d.loc[dict(level=config.nceplevels)]

        d = d.resample('MS', 'time', how='mean')
        lowres.append(d)

    cpc = read_nc_files(cpc_dir)
    cpc.load()
    cpc.time = pandas.to_datetime(cpc.time.values)
    cpc = assert_bounds(cpc, config.highres_bounds)
    cpc = cpc.resample('MS', dim='time', how='mean')
    return lowres, cpc

def read_config_data_monthly():
    import config

    vars = os.listdir(config.ncep_dir)
    lowres = []
    for v in vars:
        fv = get_ncep_file_paths(os.path.join(config.ncep_dir, v))
        if len(fv) == 0:
            continue
        print v
        d = xray.open_mfdataset(fv)
        d = assert_bounds(d, config.lowres_bounds)
        if 'level' in d.dims:
            d = d.loc[dict(level=config.nceplevels)]
            
        d.load()
        d = d.resample('MS', 'time', how='mean')
        lowres.append(d)

    cpc = read_nc_files(config.cpc_dir)
    cpc.load()
    cpc.time = pandas.to_datetime(cpc.time.values)
    cpc = assert_bounds(cpc, config.highres_bounds)
    cpc = cpc.resample('MS', dim='time', how='mean')

    return DownscaleData(lowres, cpc)



if __name__ == "__main__":

    D = read_config_data_monthly()
    print "Shape of X:", d.get_X()
    '''
    import config
    ncep_dir = config.ncep_dir
    cpc_dir = config.cpc_dir

    ncep, cpc = test_data()

    d = DownscaleData(ncep, cpc)
    X = d.get_X()

    print d.location_pairs('lat', 'lon')

    lt = d.ncep[0]['lat'].values[0]
    ln = d.ncep[0]['lon'].values[0]
    prcp = d.ncep[0].loc[{'lat':  lt, 'lon': ln}]

    from matplotlib import pyplot
    import seaborn

    pyplot.hist(X.flatten())
    pyplot.show()
    '''