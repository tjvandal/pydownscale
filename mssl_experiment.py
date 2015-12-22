__author__ = 'tj'


from pydownscale.data import DownscaleData, read_nc_files, assert_bounds
from pydownscale.downscale import DownscaleModel
from pydownscale.rMSSL import pMSSL
import time
import pydownscale.config as config
import pandas
from sklearn.linear_model import MultiTaskLassoCV, LassoCV
import numpy
import pickle

data = pickle.load(open("/scratch/vandal.t/experiments/DownscaleData/monthly_804_3150.pkl", "r"))

# print "Data Normalized: %d" % (time.time() - t0)
X = data.get_X()
y, _ = data.get_y()
season = config.seasons[0]


print "X shape", X.shape
print "Y shape", y.shape

spearmans = []
errs = []
params = []
mssl_results = []

for lambd in numpy.linspace(0, 1, 3)[1:]:
    for gamma in numpy.linspace(0, 3, 4):
        model = pMSSL(lambd=lambd, gamma=gamma, max_epochs=100)
        dmodel = DownscaleModel(data, model, season=season)
        try:
            dmodel.train()
            results = dmodel.get_results()
        except Exception as err:
            print err
            continue
        
        results = [dict(r, **{'gamma': gamma, "lambda": lambd}) for r in results]
        spear = pandas.DataFrame(results).spearman.mean()
        rmse = pandas.DataFrame(results).rmse.mean()
        spearmans.append(spear)
        errs.append(rmse)
        params.append({'gamma': gamma, "lambda": lambd})
        mssl_results.append(pandas.DataFrame(results))

print results
results = pandas.concat(mssl_results)
results.to_csv("mssl-experiment-results.csv")

idx = numpy.argmax(spearmans)
print "Max MSSL Spearman: %f" % max(spearmans) + str(params[idx]) + "\n"
print "Max MSSL Spearman:", max(spearmans), params[idx], "\n"
print "Max RMSE:", numpy.min(rmse)

pairs = data.location_pairs("lat", "lon")
lasso_results = []
for p in pairs:
    lasso = LassoCV(alphas=[0.01, 1, 10, 100])
    m1 = DownscaleModel(data, lasso, season=season)
    m1.train(location={"lat": p[0], "lon": p[1]})
    lasso_results += m1.get_results()

print "Lasso", pandas.DataFrame(lasso_results).spearman.mean(), "\n"
pandas.DataFrame(lasso_results).to_csv("lasso-results.csv")

mlasso = MultiTaskLassoCV(alphas=[0.01, 1, 10, 100])
model2 = DownscaleModel(data, mlasso, season=season)
model2.train()
results2 = model2.get_results()
print "MLLasso", pandas.DataFrame(results2).spearman.mean()
resdf = pandas.DataFrame(results2)
resdf.to_csv("mtlasso-results.csv")