#reanalysis_dir = "/gss_gpfs_scratch/vandal.t/ncep/daily/downscale-data"
reanalysis_dir = "/gss_gpfs_scratch/vandal.t/merra_2/daily_merged/"

obs_dir = "/gss_gpfs_scratch/vandal.t/cpc/regions/"
#obs_dir = "/gss_gpfs_scratch/vandal.t/gmfd/prcp/daily/0.25deg/"

gcm_dir = "/gss_gpfs_scratch/vandal.t/cmip5/historical/atm/mon/ccsm/remap/"
save_dir = "/gss_gpfs_scratch/vandal.t/experiments/"

# New England Bounds
lowres_bounds = {'lat': [35, 55], 'lon': [270, 310]}
highres_bounds = {'lat': [41, 47], 'lon': [360-74,360-67]}

#Test Small Data
#lowres_bounds = {'lat': [40, 45], 'lon': [280, 300]}
#highres_bounds = {'lat': [42, 43], 'lon': [288, 290]}

# United States Bounds
#lowres_bounds = {'lat': [20, 55], 'lon': [360-133, 310]}
#highres_bounds = {'lat': [20, 55], 'lon': [360-133, 360-50]}

train_percent = 0.67

seasons = ['DJF', 'MAM', 'JJA', 'SON']
reanalysislevels = [500, 700, 850]
gcmlevels = [50000, 70000, 85000]

reanalysisvars = ['T', 'U', 'V', 'H'] # 'TS', 'U2M', 'V2M', 'PS', 'SLP']
gcmvars = ['ta', 'ua', 'va', 'hur', 'psl']

