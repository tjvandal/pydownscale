import xray
import numpy
import os
import pandas

mssldir = "/gss_gpfs_scratch/vandal.t/experiments/mssl_nc/"
seasons = ['MAM', 'DJF']
train_ratio = 0.70
validate_ratio = 0.10

seas = seasons[1]
seasonfiles = [(f, os.path.join(mssldir, f)) for f in os.listdir(mssldir) if seas in f]
results = []
for f, fpath in seasonfiles:
    val = {}
    data = xray.open_dataset(fpath)

    n = data.time.shape[0]
    nvalidate = int(n * validate_ratio / (1 - train_ratio))
    valdata = data.loc[dict(time=data.time[:nvalidate])]
    testdata = data.loc[dict(time=data.time[nvalidate:])]
    val["rmse"] = numpy.nanmean(valdata.error.values**2)**(0.5)
    val["lmbda"] = float(f.split("_")[-1][:-3])
    val["gamma"] = float(f.split("_")[-2])
    results.append(val)

results = pandas.DataFrame(results)
res_pivot = pandas.pivot_table(results, index="lmbda", columns="gamma", values="rmse")
print res_pivot
