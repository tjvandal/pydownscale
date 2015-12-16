ncep_dir = "/scratch/vandal.t/ncep/daily/downscale-data"
cpc_dir = "/scratch/vandal.t/merged/"
save_dir = "/scratch/vandal.t/experiments/"

lowres_bounds = {'lat': [35, 55], 'lon': [270, 310]}
highres_bounds = {'lat': [41, 47], 'lon': [-74, -67]}

#lowres_bounds = {'lat': [40, 45], 'lon': [275, 300]}
#highres_bounds = {'lat': [42, 43], 'lon': [-72, -71]}


train_percent = 0.80

seasons = ['DJF', 'MAM', 'JJA', 'SON']
#nceplevels = [10, 50, 100, 250, 500, 700, 850, 1000]
nceplevels = [10, 50, 100] 
