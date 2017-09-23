#reanalysis_dir = "/gss_gpfs_scratch/vandal.t/ncep/daily/downscale-data"
reanalysis_dir = "/gss_gpfs_scratch/vandal.t/merra_2/daily_merged/"

obs_dir = "/gss_gpfs_scratch/vandal.t/cpc/regions/"
#obs_dir = "/gss_gpfs_scratch/vandal.t/gmfd/prcp/daily/0.25deg/"

gcm_dir = "/gss_gpfs_scratch/vandal.t/cmip5/historical/atm/day/ccsm4/remap/"
save_dir = "/gss_gpfs_scratch/vandal.t/sadm-experiments/"

reanalysis_londim = 'lon'
reanalysis_latdim = 'lat'

obs_londim = 'lon'
obs_latdim = 'lat'

gcm_latdim = 'lat'
gcm_londim = 'lon'

# New England Bounds
lowres_bounds = {reanalysis_latdim: [35, 55], reanalysis_londim: [270, 310]}
highres_bounds = {obs_latdim: [41, 47], obs_londim: [360-74,360-67]}

#Test Small Data
#lowres_bounds = {reanalysis_latdim: [40, 45], reanalysis_londim: [280, 300]}
#highres_bounds = {obs_latdim: [42, 43], obs_londim: [288, 290]}

# United States Bounds
#lowres_bounds = {'lat': [20, 55], 'lon': [360-133, 310]}
#highres_bounds = {'lat': [20, 55], 'lon': [360-133, 360-50]}

train_percent = 0.70
max_train_year = 2004

seasons = ['DJF', 'MAM', 'JJA', 'SON']
reanalysislevels = [500, 700, 850]
gcmlevels = [50000, 70000, 85000]

## Temperature, Horizontal wind, vertical wind, surface temperature, sea level pressure,
##   specific humidity, surface specific humidity
reanalysisvars = [ 'T', 'U', 'V', 'TS', 'SLP', 'QV', 'TQV']
gcmvars = ['ta', 'ua', 'va', 'tas', 'psl', 'hus', 'huss']

