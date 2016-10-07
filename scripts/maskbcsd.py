import xarray
import os

maskfile = "/gss_gpfs_scratch/vandal.t/cpc/masks/newengland_mask_5m.nc"
mask = xarray.open_dataset(maskfile)

asdfile = "../asd_nc/ELNET_D_DJF.nc"
asd = xarray.open_dataset(asdfile)
lats = asd.lat
lons = asd.lon

bcsdfiles = [f for f in os.listdir(".") if "_ne.nc" in f]
for f in bcsdfiles:
    print f
    season = f.split("_")[2]
    asdfile = "../asd_nc/ELNET_D_%s.nc" % season
    asd = xarray.open_dataset(asdfile)
    times = asd.time
    bcsd = xarray.open_dataset(f)
    bcsdsub = bcsd.sel(lat=lats, lon=(lons-360), time=times)
    drbcsd = xarray.DataArray(bcsdsub.projection.values, coords=[times, lats, lons], dims=['lat', 'lon', 'time']) 

    dsout = xarray.Dataset({'ground_truth': (['lat', 'lon', 'time'], asd.ground_truth.values),
                       'projected': (['time', 'lat', 'lon'], bcsdsub.projection.values)},
                      coords={'lat': lats, 'lon': lons, 'time': times}
                     )

    fout = 'BCSD_D_%s.nc' % season
    dsout['error'] = dsout['projected'] - dsout['ground_truth'] 
    dsout.to_netcdf(fout, engine='scipy')
    print "saved file"

