import xarray
import numpy
import os, sys
from qmap import QMap 
from joblib import Parallel, delayed
import pickle
import config

numpy.seterr(invalid='ignore')

def mapper(x, y, train_num):
    qmap = QMap()
    qmap.fit(x[:train_num], y[:train_num])
    return qmap.predict(y)

def nanarray(size):
    arr = numpy.empty(size)
    arr[:] = numpy.nan
    return arr

class BCSDd():
    '''
    Data in:
        Coarse GCM
        Coarse Observations
        High res Observations
    1. Bias Correction
        Compute CDF for a given day +-15 days on the coarse, retrospective, GCM 
              and coarse Observation
    2. Spatial Disaggregation
        Spatial Interpolation of Adjusted coarse GCM
    '''
    def __init__(self):
        pass

    def select_box(self, latvals, lonvals):
        latvals = latvals[numpy.where((latvals > 36) & (latvals < 52))] 
        lonvals = lonvals[numpy.where((lonvals < -60) & (lonvals > -80))]
        return latvals, lonvals

    def bias_correction(self, obs, modeled, pool=15):
        # get intersecting days
        d1 = obs.time.values
        d2 = modeled.time.values
        intersection = numpy.intersect1d(d1, d2)
        obs = obsdata.loc[dict(time=intersection)]
        modeled = modeled.loc[dict(time=intersection)]
        dayofyear = obs['time.dayofyear']
        lat_vals = modeled.lat.values
        lon_vals = modeled.lon.values
        lat_vals, lon_vals = self.select_box(lat_vals, lon_vals)
        mapped_data = numpy.zeros(shape=(intersection.shape[0], lat_vals.shape[0], lon_vals.shape[0]))
        print "Lats=%i, Lons=%i" % (len(lat_vals), len(lon_vals))
        latlonlookup = {}
        for day in numpy.unique(dayofyear.values):
            print "Day = %i" % day
            dayrange = (numpy.arange(day-pool, day+pool+1) + 366) % 366 + 1
            days = numpy.in1d(dayofyear, dayrange)
            subobs = obs.loc[dict(time=days)]
            submodeled = modeled.loc[dict(time=days)]
            sub_curr_day_rows = numpy.where(day == subobs['time.dayofyear'].values)[0]
            curr_day_rows = numpy.where(day == obs['time.dayofyear'].values)[0]
            train_num = numpy.where(subobs['time.year'] <= config.max_train_year)[0][-1]
            mapped_times = subobs['time'].values[sub_curr_day_rows]
            jobs = []
            lat_lon = []
            for lat in lat_vals:
                for lon in lon_vals:
                    if (lat, lon) not in latlonlookup.keys():
                        meshlat, meshlon = numpy.meshgrid(subobs.lat.values, subobs.lon.values)
                        dist = (meshlat - lat)**2 + (meshlon - lon)**2
                        distsort = dist.flatten().argsort()
                        latnear = meshlat.flatten()[distsort]
                        lonnear = meshlon.flatten()[distsort]
                        for lt, ln in zip(latnear, lonnear):
                            x = subobs.sel(lat=lt, lon=ln)['precip'].values
                            if not numpy.isnan(x).any():
                                latlonlookup[(lat, lon)] = (lt, ln)
                                print lt, ln
                                break
                    else:
                        lt, ln = latlonlookup[(lat, lon)]
                    x = subobs.sel(lat=lt, lon=ln)['precip'].values
                    y = submodeled.sel(lat=lat, lon=lon)['PRECTOTLAND'].values
                    jobs.append(delayed(mapper)(x, y, train_num))
                    lat_lon.append([lat, lon])

            print  "Number of pairs =", len(lat_lon)
            day_mapped = numpy.asarray(Parallel(n_jobs=48)(jobs)).T[sub_curr_day_rows]
            day_mapped = day_mapped.reshape((sub_curr_day_rows.shape[0],
                                       len(lat_vals), len(lon_vals)))

            mapped_data[curr_day_rows, :, :] = day_mapped

        dr = xarray.DataArray(mapped_data, coords=[obs['time'].values, lat_vals, lon_vals], 
                       dims=['time', 'lat', 'lon'])
        dr.attrs['gridtype'] = 'latlon'
        ds = xarray.Dataset({'bias_corrected': dr}) 
        ds = ds.reindex_like(modeled)
        modeled = modeled.merge(ds)
        del modeled['PRECTOTLAND']
        modeled.to_netcdf(os.path.join(config.save_dir, "merra_bc.nc"))

def test_bcsd():
    obs_file = "/gss_gpfs_scratch/vandal.t/cpc/merged_prcp/cpc_1980_2014.nc"
    modeled_file = "/gss_gpfs_scratch/vandal.t/merra_2/lnd/MERRA2_PRCP_D_1.0X1.25xarray.nc4"
    obs_data = xarray.open_dataset(obs_file)
    modeled_data = xarray.open_dataset(modeled_file)
    modeled_data.load()
    obs_data.load()
    obs_data = obs_data.dropna('time', how='all')

    obs_data = obs_data.resample("D", "time")
    obs_data.to_netcdf(obs_file.replace(".nc", "xarray.nc"))
    #modeled_data = modeled_data.resample("D", "time")
    #modeled_data.to_netcdf(modeled_file.replace(".nc", "xarray.nc"))


    bcsd = BCSDd()
    bcsd.bias_correction(obs_data, modeled_data)



test_bcsd()
