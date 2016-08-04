import xray
import numpy as np
import os, sys
from pydownscale.qmap import QMap
from joblib import Parallel, delayed
import pickle
from pydownscale import config

np.seterr(invalid='ignore')

def mapper(x, y, train_num):
    qmap = QMap()
    qmap.fit(x[:train_num], y[:train_num], axis=0)
    return qmap.predict(y)

def nanarray(size):
    arr = np.empty(size)
    arr[:] = np.nan
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
        latvals = latvals[np.where((latvals > 36) & (latvals < 52))] 
        lonvals = lonvals[np.where((lonvals < -60) & (lonvals > -80))]
        return latvals, lonvals

    def bias_correction(self, obs, modeled, obs_var, modeled_var, pool=15):
        # get intersecting days
        d1 = obs.time.values
        d2 = modeled.time.values
        intersection = np.intersect1d(d1, d2)
        obs = obs.loc[dict(time=intersection)]
        modeled = modeled.loc[dict(time=intersection)]
        dayofyear = obs['time.dayofyear']
        lat_vals = modeled.lat.values
        lon_vals = modeled.lon.values
        #lat_vals, lon_vals = self.select_box(lat_vals, lon_vals)
        mapped_data = np.zeros(shape=(intersection.shape[0], lat_vals.shape[0], lon_vals.shape[0]))
        print "Lats=%i, Lons=%i" % (len(lat_vals), len(lon_vals))
        latlonlookup = {}
        for day in np.unique(dayofyear.values):
            print "Day = %i" % day
            dayrange = (np.arange(day-pool, day+pool+1) + 366) % 366 + 1
            days = np.in1d(dayofyear, dayrange)
            subobs = obs.loc[dict(time=days)]
            submodeled = modeled.loc[dict(time=days)]
            sub_curr_day_rows = np.where(day == subobs['time.dayofyear'].values)[0]
            curr_day_rows = np.where(day == obs['time.dayofyear'].values)[0]
            train_num = np.where(subobs['time.year'] <= config.max_train_year)[0][-1]
            mapped_times = subobs['time'].values[sub_curr_day_rows]
            for i, lat in enumerate(lat_vals):
                jobs = []
                X_lat = subobs.sel(lat=lat, lon=lon_vals, method='nearest')[obs_var].values
                Y_lat = submodeled.sel(lat=lat, lon=lon_vals)[modeled_var].values
                jobs.append(delayed(mapper)(X_lat, Y_lat, train_num))

                print "Running jobs", len(jobs)
                day_mapped = np.asarray(Parallel(n_jobs=12)(jobs))[:, sub_curr_day_rows]
                day_mapped = np.swapaxes(day_mapped, 0, 1)
                mapped_data[curr_day_rows, i,np.newaxis, :] = day_mapped

        dr = xray.DataArray(mapped_data, coords=[obs['time'].values, lat_vals, lon_vals],
                       dims=['time', 'lat', 'lon'])
        dr.attrs['gridtype'] = 'latlon'
        ds = xray.Dataset({'bias_corrected': dr}) 
        ds = ds.reindex_like(modeled)
        modeled = modeled.merge(ds)
        del modeled[modeled_var]
        return modeled

def bias_correct_merra_prism():
    obs_file = "prism_upup_1981_2015xray.nc"
    modeled_file = "merra_filled_1981_2015xray.nc"
    obs_data = xray.open_dataset(obs_file)
    modeled_data = xray.open_dataset(modeled_file)
    # obs_data = obs_data.sel(time=(obs_data['time.year'] <= 1990))
    # modeled_data = modeled_data.sel(time=(modeled_data['time.year'] <= 1990))

    print "loading observations"
    #obs_data.load()
    #obs_data = obs_data.dropna('time', how='all')
    #obs_data = obs_data.resample("D", "time")
    #obs_data.to_netcdf(obs_file.replace(".nc", "xray.nc"))

    print "loading modeled"
    #modeled_data.load()
    #modeled_data = modeled_data.resample("D", "time")
    #modeled_data.to_netcdf(modeled_file.replace(".nc", "xray.nc"))


    print "starting bcsd"
    bc = BiasCorrectDaily()
    corrected = bc.bias_correction(obs_data, modeled_data)
    corrected.to_netcdf("merra_bc.nc")

if __name__ == "__main__":
    bias_correct_merra_prism()
