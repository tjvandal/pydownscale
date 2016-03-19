
import xray
import os, time, sys
import numpy
import pandas
from scipy.misc import factorial
import config

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
        self.reanalysis_londim = config.reanalysis_londim
        self.reanalysis_latdim = config.reanalysis_latdim
        self.obs_londim = config.obs_londim
        self.obs_latdim = config.obs_latdim

    def _check_reanalysis(self):
		for _, n in self.reanalysis.iteritems():
			if (not isinstance(n, xray.Dataset)) and (not isinstance(n, xray.DataArray)):
				raise TypeError("Reanalysis Data should be a list of xray datasets or dataarrays.")

	# can we make this dynamic in the future?
    def _checkindices(self):
        times = []
        for _, d in self.reanalysis.iteritems():
            if not 'time' in d.dims:
                raise IndexError("time should be a dimension of cmip")

        if not 'time' in self.observations.dims:
            raise IndexError("time should be a dimension of observations")

	# Line up datasets so that the times match.  Exclude observations outside of the timeframe.
    def _matchdates(self):
        obstime = self.observations.time.values
        reanalysistimes = []
        for var in config.reanalysisvars:
            reanalysistimes.append(self.reanalysis[var].time.values)
        timeset = obstime
        for j, val in enumerate(reanalysistimes):
            timeset = numpy.intersect1d(timeset, val)
        self.observations = self.observations.loc[{'time': timeset}]
        for j, var in enumerate(config.reanalysisvars):
            self.reanalysis[var] = self.reanalysis[var].loc[{'time': timeset}]

    def get_X(self, timedim='time'):
        import config
        x = []
        for var in config.reanalysisvars:
            self.reanalysis[var].load()
            df = self.reanalysis[var].to_array().to_dataframe()
            levels = sorted([v for v in df.index.names if v not in (timedim, 'bnds')])
            x.append(df.unstack(levels).values)
        x = numpy.column_stack(x)
        return x

    def get_XTensor(self, timedim='time'):
        XT = []
        for var in self.reanalysis:
            xv = self.reanalysis[var][var].values
            if len(xv.shape)==4:
                xv = numpy.swapaxes(xv, 1, 3)
                xv = numpy.swapaxes(xv, 1, 2)
            elif len(xv.shape) == 3:
                xv = xv[:, :, :, numpy.newaxis]
            XT.append(xv)

        XT = numpy.concatenate(XT, axis=3)
        return XT

    def get_nearest_X(self, latval, lonval, timedim='time'):
        import config
        x = []
        for var in config.reanalysisvars:
            self.reanalysis[var].load()
            lats = self.reanalysis[var][self.reanalysis_latdim].values
            lons = self.reanalysis[var][self.reanalysis_londim].values
            latbelow = numpy.where(lats < latval)[0][-2]
            latabove = numpy.where(lats > latval)[0][1]
            lonbelow = numpy.where(lons < lonval)[0][-2]
            lonabove = numpy.where(lons > lonval)[0][1]
            subset = self.reanalysis[var][{self.reanalysis_latdim: slice(latbelow, latabove+1), 
                                           self.reanalysis_londim: slice(lonbelow, lonabove+1)}]
            df = subset.to_array().to_dataframe()
            levels = sorted([v for v in df.index.names if v not in (timedim, 'bnds')])
            x.append(df.unstack(levels).values)
        x = numpy.column_stack(x)
        return x

    def get_y(self, location=None, timedim='time'):
        if location is not None:
            y = self.observations.loc[location].to_array().values.squeeze()
        else:
            y = self.observations.to_array().to_dataframe()
            levels =sorted([var for var in y.index.names if var != timedim])
            y = y.unstack(levels)
            location = y.columns.to_series()
            y = y.values
            ycolmean = y.mean(axis=0)
            ybelow = (y >= 0).mean(axis=0)
            idx = numpy.where(ybelow != 1)[0][-1]
            idx2 = numpy.where(y[:, idx] == -999.)[0]
            cols = (ycolmean !=0 ) * (ybelow == 1.) * (~numpy.isnan(ycolmean))
            y = y[:, cols]
            location = location[cols]
            names = location.index.names
            dim1idx = names.index(self.obs_latdim)
            dim2idx = names.index(self.obs_londim)
            locs = [[row[dim1idx], row[dim2idx]] for row in location.values]
            location = pandas.DataFrame(numpy.vstack(locs), columns=[self.obs_latdim, self.obs_londim])
        return y, location

    def location_pairs(self, dim1, dim2):
        Y = self.observations.to_array()
        if (dim1 not in Y.dims) or (dim2 not in Y.dims):
            raise IndexError("dim1=%s and dim2=%s are not in observations." % (dim1, dim2))
        if len(Y.dims) != 4:
            raise ValueError("There should be 4 dimensions with only 1 variable.")

        t0 = Y['time'].values[0]
        t0array = self.observations.loc[dict(time=t0)].to_array()
        t0array = self.observations.mean(dim='time').to_array()

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

        pairs = [[d1, d2] for i1, d1 in enumerate(Y.coords[dim1].values) for i2, d2 in
                 enumerate(Y.coords[dim2].values) if t0values[i1, i2] not in (-999., numpy.nan, 0.)]
        return pairs

class GCMData:
    def __init__(self, data):
        self.data = data
        self._matchdates()

    def _matchdates(self):
        times = []
        for var in config.gcmvars:
            times.append(self.data[var].time.values)
        timeset = times[0]
        for t in times[1:]:
            timeset = numpy.intersect1d(timeset, t)
        for j, var in enumerate(config.gcmvars):
            tvar = numpy.in1d(times[j], timeset)
            self.data[var] = self.data[var].loc[{'time': timeset}]

    def get_X(self, timedim='time', season=None):
        x = []
        key0 = config.gcmvars[0]
        t = self.data[key0][timedim].values
        for var in config.gcmvars:
            self.data[var].load()
            df = self.data[var].to_array().to_dataframe()
            levels = sorted([v for v in df.index.names if v != timedim])
            x.append(df.unstack(levels).values)
        x = numpy.column_stack(x)
        if season is not None:
            seasonidxs = numpy.where(self.data[key0]['time.season']== season)[0]
            x = x[seasonidxs, :]
            t = t[seasonidxs]
        return x, t

    def get_nearest_X(self, latval, lonval, season=None, timedim='time'):
        x = []
        key0 = config.gcmvars[0]
        t = self.data[key0][timedim].values
        for var in config.gcmvars:
            self.data[var].load()
            lats = self.data[var][config.gcm_latdim].values
            lons = self.data[var][config.gcm_londim].values
            latbelow = numpy.where(lats < latval)[0][-2]
            latabove = numpy.where(lats > latval)[0][1]
            lonbelow = numpy.where(lons < lonval)[0][-2]
            lonabove = numpy.where(lons > lonval)[0][1]
            subset = self.data[var][{config.gcm_latdim: slice(latbelow, latabove+1), 
                                           config.gcm_londim: slice(lonbelow, lonabove+1)}]
            df = subset.to_array().to_dataframe()
            levels = sorted([v for v in df.index.names if v not in (timedim, 'bnds')])
            x.append(df.unstack(levels).values)
        x = numpy.column_stack(x)
        if season is not None:
            seasonidxs = numpy.where(self.data[key0]['time.season']== season)[0]
            x = x[seasonidxs, :]
            t = t[seasonidxs]
        return x, t

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
        data = xray.open_mfdataset(files, preprocess=lambda d: assert_bounds(d, bounds))
    elif len(files) == 1:
        data = xray.open_mfdataset(files, preprocess=lambda d: assert_bounds(d, bounds))
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
        baselevels = config.reanalysislevels
    elif which == 'gcm':
        vars = config.gcmvars
        basedir = config.gcm_dir
        baselevels = config.gcmlevels

    lowres = {} 
    for v in vars:
        fv = get_reanalysis_file_paths(os.path.join(basedir, v))
        if len(fv) == 0:
            continue

        d = xray.open_mfdataset(fv, preprocess=lambda d: assert_bounds(d, config.lowres_bounds))
        d.load()
        levs = set(d.dims.keys()).intersection(set(('lev', 'plev', 'level')))
        if len(levs) >  1:
            raise(Exception("What level to use?" + str(levs)))
        elif len(levs) == 0:
            lowres[v] = d
        else:
            lev = levs.pop()
            levels = [l for l in baselevels if l in d[lev].values]
            if len(levels) > 0:
                d = d.loc[{lev: levels}]

            d.rename({lev: 'plev'}, inplace=True)

        if 'time_bnds' in d.keys():
            del d['time_bnds']
        if 'bnds' in d.keys():
            del d['bnds']
        d = d.resample(how, 'time', how='mean')
        print v, "Before Drop", d[v].size
        d = d.dropna('time')
        print v, "After Drop", d[v].size
        lowres[v] = d

    return lowres

def read_obs(how='MS'):
    import config
    obs = read_nc_files(config.obs_dir, config.highres_bounds)
    obs.load()
    obs.time = pandas.to_datetime(obs.time.values)
    obs = obs.resample(how, dim='time', how=numpy.mean)
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
    how = "D"
    print "reading reanalysis"
    reanalysis = read_lowres_data(which='reanalysis', how=how)
    print "reading observations"
    obs = read_obs(how='D')  # we are in mm
    rows = ((obs == -999).mean(dim=('lat', 'lon'))['precip']) < 0.20    # delete those with lots of 999.s 
    times = obs['time'][rows]
    print "time skipped", obs['time'][~rows]
    obs = obs.loc[{'time': times}]
    #obs = obs.resample(how, dim='time', how='mean')

    D = DownscaleData(reanalysis, obs)
    X = D.get_X()

    fname = "newengland_%s_%i_%i.pkl" % (how, X.shape[0], X.shape[1])
    f = os.path.join(config.save_dir, "DownscaleData", fname)
    pickle.dump(D, open(f, 'w'))
    y, loc = D.get_y()

    print "Number of Tasks:", len(D.location_pairs("lat", "lon"))
    print "Shape of X:", X.shape
    print "Saved for file: %s" % f
    '''
    gcmf = "/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/ccsm4_historical_%s.pkl" % how
    #G = pickle.load(open(gcmf, "r"))
    gcm = read_lowres_data(which='gcm', how=how)
    G = GCMData(gcm)
    pickle.dump(G, open(gcmf, "w"))
    GX = G.get_X()
    print "SHAPE OF GCM X:", GX.shape
    GXN = G.get_nearest_X(42.125, 360-71)
    print "Shape of nearest:", GXN.shape
    '''
