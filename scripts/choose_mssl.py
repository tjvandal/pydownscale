import xray
import numpy
import os
import pandas

mssldir = "/gss_gpfs_scratch/vandal.t/experiments/mssl_nc/"
seasons = ['JJA', 'MAM', 'DJF', 'SON']
train_year = 1999
validate_year = 2004

seas = seasons[0]
for seas in seasons:
    seasonfiles = [(f, os.path.join(mssldir, f)) for f in os.listdir(mssldir) if seas in f]
    results = []
    for f, fpath in seasonfiles:
        val = {}
        data = xray.open_dataset(fpath)
        data.load()
        n = data.time.shape[0]
        valrows = numpy.where((data['time.year'] > train_year) & (data['time.year'] <= validate_year))[0] 
        testrows = numpy.where(data['time.year'] > validate_year)[0]
        valtimes = data.time[valrows]
        testtimes = data.time[testrows]
        valdata = data.loc[dict(time=valtimes)]
        testdata = data.loc[dict(time=testtimes)]
        val["rmse"] = numpy.nanmean(valdata.error.values**2)**(0.5)
        val["lmbda"] = float(f.split("_")[-1][:-3])
        val["gamma"] = float(f.split("_")[-2])
        results.append(val)

    results = pandas.DataFrame(results)
    res_pivot = pandas.pivot_table(results, index="lmbda", columns="gamma", values="rmse")
    results.sort('rmse', inplace=True)
    print "Season", seas
    print results.head()
