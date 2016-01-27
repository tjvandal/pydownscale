from data import DownscaleData, read_nc_files, assert_bounds
import numpy
from scipy.stats import pearsonr, spearmanr, kendalltau
import config
import pandas
import pickle 
import time

class DownscaleModel:
    def __init__(self, data, model, training_size=config.train_percent, season=None, feature_columns=None):
        if data.__class__.__name__ != 'DownscaleData':
            raise TypeError("Data must be of type downscale data.")
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("Model must have methods train and predict.")

        self.data = data
        self.model = model
        self.season = season
        if self.season:
            key0 = self.data.reanalysis.keys()[0]
            self.seasonidxs = numpy.where(self.data.reanalysis[key0]['time.season'] == self.season)[0]

        self._split_dataset(training_size, feature_columns=feature_columns)

    def _split_dataset(self, training_size, feature_columns=None):
        if feature_columns is not None:
            X = self.data.get_X()[:, feature_columns]
        else:
            X = self.data.get_X()

        if self.season:
            X = X[self.seasonidxs, :]

        self.numtrain = int(X.shape[0] * training_size)
        self.X_train = X[:self.numtrain]
        self.X_test = X[self.numtrain:]

    # include quantile mapping ability
    def train(self, location=None):
        start_time = time.time()
        y, self.location = self.data.get_y(location=location)
        t = self.data.observations['time'].values
        if self.season:
            y = y[self.seasonidxs]
            t = t[self.seasonidxs]

        self.y_train = y[:self.numtrain]
        self.y_test = y[self.numtrain:]
        self.model.fit(self.X_train, self.y_train)
        self.yhat_train = self.model.predict(self.X_train)
        self.yhat_test = self.model.predict(self.X_test)
        self.t_test = t[self.numtrain:]

    def get_mse(self, y, yhat):
        return numpy.mean((y - yhat) ** 2)

    def _stats(self, y, yhat, test_set=True):
        res = {}
        res['rmse'] = self.get_mse(y, yhat)**(0.5)
        res['pearson'] = pearsonr(y, yhat)[0]
        res['spearman'] = spearmanr(y, yhat)[0]
        res['kendaltau'] = kendalltau(y, yhat)[0]
        res['yhat_mean'] = numpy.mean(yhat)
        res['y_mean'] = numpy.mean(y)
        res['yhat_std'] = numpy.std(yhat)
        res['y_std'] = numpy.std(y)
        res['model_name'] = self.model.__class__.__name__
        res['season'] = self.season
        return res

    # add functionality: plotting, quantile mapping, distirbutions, correlations
    def get_results(self, test_set=True):
        if test_set:
            y = self.y_test
            yhat = self.yhat_test
        else:
            y = self.y_train
            yhat = self.yhat_train

        results = []
        if isinstance(self.location, dict):
            self.location.update(self._stats(y, yhat, test_set))
            self.location['y'] = y
            self.location['yhat'] = yhat
            self.location['time'] = self.t_test
            results.append(self.location)

        elif isinstance(self.location, pandas.Series):
            names = self.location.index.names
            for j, row in enumerate(self.location.iteritems()):
                r = row[1]
                res = {n: r[i] for i, n in enumerate(names)}
                res.update(self._stats(y[:, j], yhat[:, j], test_set=test_set))
                res['y'] = y[:, j]
                res['yhat'] = yhat[:, j]
                res['time'] = self.t_test

                if hasattr(self.model, "alpha_"):
                    res["lasso_alpha"] = numpy.mean(self.model.alpha_)
                results.append(res)
            pass

        return results

    def predict(self, X):
        pass


if __name__ == "__main__":
    import pandas
    from sklearn.linear_model import LassoCV, LinearRegression
    import time
    t0 = time.time()

    cmip5_dir = "/Users/tj/data/reanalysis_ncar_monthly/"
    cpc_dir = "/Users/tj/data/usa_cpc_nc/merged/"

    # climate model data, monthly
    cmip5 = read_nc_files(cmip5_dir)
    cmip5.load()
    cmip5 = assert_bounds(cmip5, {'lat': [15, 55], 'lon': [200, 320]})
    cmip5 = cmip5.resample('MS', 'time', how='mean')   ## try to not resample

    # daily data to monthly
    cpc = read_nc_files(cpc_dir)
    cpc.load()
    cpc = assert_bounds(cpc, {'lat': [42, 42.5], 'lon': [-71, -70]})
    monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

    print "Data Loaded: %d seconds" % (time.time() - t0)
    data = DownscaleData(cmip5, monthlycpc)
    data.normalize_monthly()
    pairs = data.location_pairs('lat', 'lon')

    # print "Data Normalized: %d" % (time.time() - t0)
    linearmodel = LassoCV(alphas=[0.01, 0.1, 1, 10, 100], max_iter=2000)
    #linearmodel = LinearRegression()
    dmodel = DownscaleModel(data, linearmodel, season='DJF') #, xvars=['uas', 'vas', 'tasmax', 'tasmin', 'hurs'])
    dmodel.train(location={'lat': pairs[0][0], 'lon': pairs[0][1]})
    X = dmodel.X_train
    y = dmodel.y_train
    print dmodel.get_results()

    from matplotlib import pyplot
    import seaborn

    pyplot.plot(dmodel.y_test)
    pyplot.plot(dmodel.yhat_test)
    pyplot.show()
    nonzero = numpy.where(dmodel.model.coef_ != 0)[0]
    print "chosen alpha", dmodel.model.alpha_
    print "NON zero indicies", nonzero, sum(numpy.abs(dmodel.model.coef_))
    print X[:, nonzero].shape

    print "Time to downscale: %d" % (time.time() - t0)
