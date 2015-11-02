__author__ = 'tj'


from pydownscale.data import DownscaleData, read_nc_files, assert_bounds
from pydownscale.downscale import DownscaleModel
from pydownscale.rMSSL import pMSSL
import time
import pydownscale.config as config
import pandas
from sklearn.linear_model import MultiTaskLassoCV, LassoCV
import numpy

t0 = time.time()
cmip5_dir = config.cmip5_dir
cpc_dir = config.cpc_dir

# climate model data, monthly
cmip5 = read_nc_files(cmip5_dir)
cmip5.load()
cmip5 = assert_bounds(cmip5, config.lowres_bounds)
cmip5 = cmip5.resample('MS', 'time', how='mean')   ## try to not resample

# daily data to monthly
cpc = read_nc_files(cpc_dir)
cpc.load()
cpc = assert_bounds(cpc, config.highres_bounds)
monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

print "Data Loaded: %d seconds" % (time.time() - t0)
data = DownscaleData(cmip5, monthlycpc)
data.normalize_monthly()

# print "Data Normalized: %d" % (time.time() - t0)
X = data.get_X()
y, _ = data.get_y()
season = config.seasons[0]


print "X shape", X.shape
print "Y shape", y.shape

spearmans = []
params = []
for lambd in [1e-2, 1e0, 1e1, 1e2][:1]:
    for gamma in [1e1, 1e0, 1e1, 1e2][:1]:
        model = pMSSL(lambd=lambd, gamma=gamma)
        dmodel = DownscaleModel(data, model, season=season)
        dmodel.train()
        results = dmodel.get_results()
        spear = pandas.DataFrame(results).spearman.mean()
        spearmans.append(spear)
        params.append({'gamma': gamma, "lambda": lambd})
        print spear

idx = numpy.argmax(spearmans)
print "Max MSSL Spearman:", max(spearmans), params[idx], "\n"

pairs = data.location_pairs("lat", "lon")
lasso_results = []
for p in pairs:
    lasso = LassoCV(alphas=[0.01, 1, 10, 100])
    m1 = DownscaleModel(data, lasso, season=season)
    m1.train(location={"lat": p[0], "lon": p[1]})
    lasso_results += m1.get_results()

print "Lasso", pandas.DataFrame(lasso_results).spearman.mean(), "\n"

mlasso = MultiTaskLassoCV(alphas=[0.01, 1, 10, 100])
model2 = DownscaleModel(data, mlasso, season=season)
model2.train()
results2 = model2.get_results()
print "MLLasso", pandas.DataFrame(results2).spearman.mean()