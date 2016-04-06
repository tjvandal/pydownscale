import matplotlib
matplotlib.use('Agg')

import xray
import numpy
import os
import pandas
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


def plot_3d(x, y, z, ax=None):
    if ax is None:
        fig = plt.figure()
        ax = fig.gca(projection='3d')
    print x, y
    x, y = numpy.meshgrid(x, y)
    surf = ax.plot_wireframe(x, y, z)


mssldir = "/gss_gpfs_scratch/vandal.t/experiments/msslerr_nc/"
seasons = ['JJA', 'MAM', 'DJF', 'SON']
train_year = 1999
validate_year = 2004

seas = seasons[0]
for seas in seasons[:2]:
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
        val["rmse_val"] = numpy.nanmean(valdata.error.values**2)**(0.5)
        val["rmse_test"] = numpy.nanmean(testdata.error.values**2)**(0.5)
        val["lmbda"] = float(f.split("_")[-1][:-3])
        val["gamma"] = float(f.split("_")[-2])
        results.append(val)

    results = pandas.DataFrame(results)
    res_pivot = pandas.pivot_table(results, index="lmbda", columns="gamma", values="rmse_test")
    res_pivot = res_pivot.interpolate()
    results.sort('rmse_test', inplace=True)
    print "Season", seas
    print res_pivot
    plot_3d(numpy.log(res_pivot.columns), numpy.log(res_pivot.index), res_pivot.values/10.)
    plt.savefig("figures/%s_MSSL_RMSE.pdf" % seas)

    lmbdarmse = res_pivot[0.001]
    plt.close()
    plt.semilogx(lmbdarmse.index, lmbdarmse.values)
    plt.savefig("figures/%s_lmdarmse.pdf" % seas)
