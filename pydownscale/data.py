
import xray
import os, time, sys
import numpy
import pandas
from scipy.misc import factorial

'''
Lets let reanalysis be a list of xray datasets where the time axis matches but may have different levels
normalization is no needed at this step, atleast until training
'''

class DownscaleData:
    def __init__(self, reanalysis, observations):
		if isinstance(reanalysis, xray.Dataset):
			reanalysis = [reanalysis]

		self.reanalysis = reanalysis
		self.observations = observations
		self._check_reanalysis()
		self._checkindices()
		self._matchdates()

    def _check_reanalysis(self):
		for _, n in self.reanalysis.iteritems():
			if (not isinstance(n, xray.Dataset)) and (not isinstance(n, xray.DataArray)):
				raise TypeError("Ncep should be a list of xray datasets or dataarrays.")

	# can we make this dynamic in the future?
    def _checkindices(self):
		times = []
		for _, d in self.reanalysis.iteritems():
			if not 'time' in d.dims:
				raise IndexError("time should be a dimension of cmip")
			times.append(d.time.values)

		times = numpy.array(times)
		if numpy.sum(times[0] - times).item() != 0:
			raise IndexError("All times in lowres data should be identical.")

		if not 'time' in self.observations.dims:
			raise IndexError("time should be a dimension of observations")

	# Line up datasets so that the times match.  Exclude observations outside of the timeframe.
    def _matchdates(self):
		key1 = self.reanalysis.keys()[0]
		mintime = max(self.reanalysis[key1].time.min(), self.observations.time.min())
		maxtime = min(self.reanalysis[key1].time.max(), self.observations.time.max())

		cmiptimes = (self.reanalysis[key1].time >= mintime) & (self.reanalysis[key1].time <= maxtime)
		obstimes = (self.observations.time >= mintime) & (self.observations.time <= maxtime)

		for key in self.reanalysis:
			self.reanalysis[key] = self.reanalysis[key].loc[dict(time=self.reanalysis[key].time[cmiptimes])]

		self.observations = self.observations.loc[dict(time=self.observations.time[obstimes])]

		if (len(self.reanalysis[key1].time) != len(self.observations.time)) or \
				sum(self.reanalysis[key1].time == self.observations.time) != len(self.reanalysis[key1].time):
				print self.reanalysis[key1].time, self.observations.time
				raise IndexError("times do not match.  add functionality if this is not an error")

    def get_X(self, timedim='time'):
        import config
        x = []
        for var in config.reanalysisvars:
            self.reanalysis[var].load()
            print self.reanalysis[var]
            df = self.reanalysis[var].to_array().to_dataframe()
            levels = sorted([v for v in df.index.names if v not in (timedim, 'bnds')])
            x.append(df.unstack(levels).values)
        x = numpy.column_stack(x)
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
			cols = y[0,:] != -999.
			y = y[:, cols]
			location = location[cols]
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

		pairs = [[d1, d2] for i1, d1 in enumerate(Y.coords[dim1].values) for i2, d2 in enumerate(Y.coords[dim2].values) if t0values[i1, i2] not in (-999., numpy.nan)]
		return pairs

class GCMData:
	def __init__(self, data):
		self.data = data		

	def get_X(self, timedim='time'):
		import config
		x = []
		for var in config.gcmvars:
			self.data[var].load()
			df = self.data[var].to_array().to_dataframe()
			levels = sorted([v for v in df.index.names if v != timedim])
			x.append(df.unstack(levels).values)

		x = numpy.column_stack(x)
		return x

def get_reanalysis_file_paths(basedir):
    files = []
    for (dirpath, dirnames, filenames) in os.walk(basedir):
        for f in filenames:
            if '.nc' in f:
                files.append(os.path.join(dirpath, f))

    return sorted(files)

def read_nc_files(dir, bounds=None):
    def rmheight(d):
        #del d["height"]
        return d

    files = get_reanalysis_file_paths(dir)
    if len(files) > 1:
        data = xray.open_mfdataset(files[30:33], preprocess=lambda d: assert_bounds(d, bounds))
    elif len(files) == 1:
        data = xray.open_dataset(files[0])
    else:
        raise IOError("There are no .nc files in that directory.")
    return data

def assert_bounds(data, bounds=None):
    '''
    :param data: Should be an xray dataset
    :param bounds: dictionary of bounds {'lat': [l1 ,l2], 'lon': [l3, l4]}
    :return: xray dataset bounded by the bounds
    '''
    def convert_lon(val):
        return (val + 360) % 360

    loc = {}
    if bounds is None:
        return data

    for b in bounds:
        vals = data[b].values
        if b in ('lon', 'longitude'):
            data[b] = convert_lon(vals)
            vals = data[b].values

        loc[b] = vals[(bounds[b][0] < vals) & (bounds[b][1] > vals)]
    return data.loc[loc]

def test_data():
    p1 = "/scratch/vandal.t/reanalysis/daily/downscale-data/air/"
    p2 = "/scratch/vandal.t/reanalysis/daily/downscale-data/pr_wtr/"
    files1 = get_reanalysis_file_paths(p1)[2:3]
    files2 = get_reanalysis_file_paths(p2)[2:3]

    d1 = xray.open_mfdataset(files1)
    d2 = xray.open_mfdataset(files2)
    lowres = []
    for d in [d1, d2]:
        d.load()
        d = assert_bounds(d, config.lowres_bounds)
        if 'level' in d.dims:
            d = d.loc[dict(level=config.reanalysislevels)]

        d = d.resample('MS', 'time', how='mean')
        lowres.append(d)

    obs = read_nc_files(obs_dir)
    obs.load()
    obs.time = pandas.to_datetime(obs.time.values)
    obs = assert_bounds(obs, config.highres_bounds)
    obs = obs.resample('MS', dim='time', how='mean')
    return lowres, obs

def read_lowres_data(how='MS', which='reanalysis'):
    import config
    if which == 'reanalysis':
        vars = config.reanalysisvars
        basedir = config.reanalysis_dir
    elif which == 'gcm':
        vars = config.gcmvars
        basedir = config.gcm_dir

    lowres = {} 
    for v in vars:
        fv = get_reanalysis_file_paths(os.path.join(basedir, v))
        if len(fv) == 0:
            continue

        d = xray.open_mfdataset(fv[:3], preprocess=lambda d: assert_bounds(d, config.lowres_bounds))
        print "loading"
        d.load()
        levs = set(d.dims.keys()).intersection(set(('lev', 'plev', 'level')))
        if len(levs) >  1:
            raise(Exception("What level to use?" + str(levs)))
        elif len(levs) == 0:
            lowres[v] = d
            continue
        else:
            lev = levs.pop()
        
        levels = [l for l in config.reanalysislevels if l in d[lev].values]
        if len(levels) > 0:
            d = d.loc[{lev: levels}]

        d.rename({lev: 'plev'}, inplace=True)

        if 'time_bnds' in d.keys():
            del d['time_bnds']
        if 'bnds' in d.keys():
            del d['bnds']

        d = d.resample(how, 'time', how='mean')
        lowres[v] = d

    return lowres

def read_obs(how='MS'):
    import config
    obs = read_nc_files(config.obs_dir, config.highres_bounds)
    obs.load()
    obs.time = pandas.to_datetime(obs.time.values)
    obs = obs.resample('MS', dim='time', how='mean')
    return obs

def test_transformation():
    import config
    import pickle
    
    variables = zip(config.reanalysisvars, config.gcmvars)
    print variables
    print "reading reanalysis"
    reanalysis = read_lowres_data(which='reanalysis', how='MS')
    print "reading observations"
    obs = read_obs(how='obs')
    D = DownscaleData(reanalysis, obs)
    gcm = read_lowres_data(which='gcm', how='MS')
    G = GCMData(gcm)
    print G.data.keys()
    for i, (vr, vg) in enumerate(variables):
        print i, vr, vg
    X = D.get_X()
    print "Shape of X:", X.shape

    GX = G.get_X()

if __name__ == "__main__":
    import config
    import pickle
    
#    test_transformation()
#    sys.exit()

    print "reading reanalysis"
    reanalysis = read_lowres_data(which='reanalysis', how='MS')
    print "reading observations"
    obs = read_obs(how='obs')
    D = DownscaleData(reanalysis, obs)
    X = D.get_X()
    print "Shape of X:", X.shape
    fname = "monthly_%i_%i.pkl" % (X.shape[0], X.shape[1])
    f = os.path.join(config.save_dir, "DownscaleData", fname)
    pickle.dump(D, open(f, 'w'))
    print "Saved for file: %s" % f

    gcm = read_lowres_data(which='gcm', how='MS')
    G = GCMData(gcm)
    GX = G.get_X()
    print "SHAPE OF GCM X:", GX.shape

    '''
    Daily = read_config_data_daily()
    X = Daily.get_X()
    print "Shape of X:", X.shape
    fname = "daily_%i_%i.pkl" % (X.shape[0], X.shape[1])
    f = os.path.join(config.save_dir, "DownscaleData", fname)
    pickle.dump(Daily, open(f, 'w'))
    print "Saved for file: %s" % f
    '''

    '''
    import config
    reanalysis_dir = config.reanalysis_dir
    obs_dir = config.obs_dir

    reanalysis, obs = test_data()

    d = DownscaleData(reanalysis, obs)
    X = d.get_X()

    print d.location_pairs('lat', 'lon')

    lt = d.reanalysis[0]['lat'].values[0]
    ln = d.reanalysis[0]['lon'].values[0]
    prcp = d.reanalysis[0].loc[{'lat':  lt, 'lon': ln}]

    from matplotlib import pyplot
    import seaborn

    pyplot.hist(X.flatten())
    pyplot.show()
    '''
