from data import DownscaleData, read_nc_files, assert_bounds
import numpy
from scipy.stats import pearsonr, spearmanr, kendalltau
import config
import pandas
import pickle 
import time
from stepwise_regression import BackwardStepwiseRegression
from joblib import Parallel, delayed
import copy
import xray
import utils
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LassoCV

class DownscaleModel:
    def __init__(self, data, model, training_size=config.train_percent, season=None, feature_columns=None):
        if data.__class__.__name__ != 'DownscaleData':
            raise TypeError("Data must be of type downscale data.")
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("Model must have methods train and predict.")

        self.data = data
        self.model = model
        self.season = season
        self.feature_columns = feature_columns
        self.training_size = training_size
        if self.season:
            key0 = self.data.reanalysis.keys()[0]
            self.seasonidxs = numpy.where(self.data.reanalysis[key0]['time.season'] == self.season)[0]


    def _split_dataset(self):
        if self.feature_columns is not None:
            X = self.data.get_X()[:, self.feature_columns]
        else:
            X = self.data.get_X()

        if self.season:
            X = X[self.seasonidxs, :]

        self.numtrain = int(X.shape[0] * self.training_size)
        self.X_train = X[:self.numtrain]
        self.X_test = X[self.numtrain:]


    # include quantile mapping ability
    def train(self, location=None):
        self._split_dataset()
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

        return results

    def predict(self, X):
        pass

def asd_worker(model, X, y):
    model.fit(X, y)
    return model

class ASD(DownscaleModel):
    def __init__(self, data, model=BackwardStepwiseRegression(), training_size=config.train_percent,
                 season=None, feature_columns=None, latdim='lat', londim='lon',
                 nearest_neighbor=True, ytransform=None, xtransform=None, num_proc=48):
        DownscaleModel.__init__(self, data, model, training_size=training_size, season=season)
        self.nearest_neighbor = nearest_neighbor
        self.xtransform = xtransform
        self.ytransform = ytransform
        self.num_proc = num_proc

    def _split_dataset(self, X, test_set=False):
        if self.season:
            X = X[self.seasonidxs]
        idx = int(X.shape[0] * self.training_size)
        if test_set:
            return X[idx:]
        else:
            return X[:idx]

    def train(self, location=None):
        y, self.locations = self.data.get_y(location=location)
        jobs = []
        if self.xtransform is not None:
            self.xtrans = []
        if self.ytransform is not None:
            self.ytrans = []
        if not self.nearest_neighbor:
            X = self.data.get_X()
            Xtrain = self._split_dataset(X)
            if self.xtransform is not None:
                self.xtrans = self.xtransform
                self.xtrans.fit(Xtrain)
                Xtrain = self.xtrans.transform(Xtrain)
        for j, row in self.locations.iterrows():
            if self.nearest_neighbor:
                X = self.data.get_nearest_X(row[self.data.reanalysis_latdim],
                                   row[self.data.reanalysis_londim])
                Xtrain = self._split_dataset(X)
                if self.xtransform is not None:
                    self.xtrans += [copy.deepcopy(self.xtransform)]
                    self.xtrans[j].fit(Xtrain)
                    Xtrain = self.xtrans[j].transform(Xtrain)

            ytrain = self._split_dataset(y[:,j])
            ytrain = ytrain.reshape(-1,1)

            if self.ytransform is not None:
                self.ytrans += [copy.deepcopy(self.ytransform)]
                self.ytrans[j].fit(ytrain)
                ytrain = self.ytrans[j].transform(ytrain)
            jobs.append(delayed(asd_worker)(copy.deepcopy(self.model), Xtrain,
                                            ytrain.flatten()))
        self.models = Parallel(n_jobs=self.num_proc)(jobs)

    def predict(self, test_set=True, location=None):
        Y, self.locations = self.data.get_y(location=location)
        t = self.data.observations['time'].values
        t = self._split_dataset(t,test_set=test_set)
        Y = self._split_dataset(Y, test_set=test_set)
        yhat = []
        ytrue =[]
        if not self.nearest_neighbor:
            X = self.data.get_X()
            X = self._split_dataset(X, test_set=test_set) 
            if self.xtransform is not None:
                X = self.xtrans.transform(X)
        for j, row in self.locations.iterrows():
            if self.nearest_neighbor:
                X = self.data.get_nearest_X(row[self.data.reanalysis_latdim],
                                   row[self.data.reanalysis_londim])

                X = self._split_dataset(X, test_set=test_set) 
                if self.xtransform is not None:
                    X = self.xtrans[j].transform(X)

            yhat += [self.models[j].predict(X)]
            #print "Yhat range", numpy.min(yhat[-1]), numpy.max(yhat[-1])
            if self.ytransform is not None:
                yhat[j] = self.ytrans[j].inverse_transform(yhat[j])
            ytrue += [Y[:, j]]
        yhat = numpy.vstack(yhat).T
        ytrue = numpy.vstack(ytrue).T
        yhat = self.to_xray(yhat, t).rename({"value": "projected"})
        ytrue = self.to_xray(ytrue, t).rename({"value": "ground_truth"})
        out = yhat.merge(ytrue)
        out['error'] = out.projected - out.ground_truth
        return out

    def to_xray(self, y, times):
        data = []
        for i, row in self.locations.iterrows():
            for j, t in enumerate(times):
                data.append({"value": y[j,i], self.data.obs_latdim: row[self.data.obs_latdim],
                             'time': t, self.data.obs_londim: row[self.data.obs_londim]})
        data = pandas.DataFrame(data)
        data.set_index([self.data.obs_latdim, self.data.obs_londim, "time"], inplace=True)
        data = xray.Dataset.from_dataframe(data)
        return data

    def project_gcm(self, gcm):
        yhat = []
        X, t = gcm.get_X(season=self.season)
        nanrows = numpy.any(numpy.isnan(X), axis=1)
        X = X[~nanrows]
        t = t[~nanrows] 
        if (not self.nearest_neighbor) and (self.xtransform is not None):
            X = self.xtrans.transform(X)
        for j, row in self.locations.iterrows():
            if self.nearest_neighbor:
                X, t = gcm.get_nearest_X(row[self.data.reanalysis_latdim],
                                      row[self.data.reanalysis_londim], season=self.season)
                X = X[~nanrows]
                if self.xtransform is not None:
                    X = self.xtrans[j].transform(X)
            yhat += [self.models[j].predict(X)]
            if self.ytransform is not None:
                yhat[j] = self.ytrans[j].inverse_transform(yhat[j])
        yhat = numpy.vstack(yhat).T
        yhat = self.to_xray(yhat, t).rename(dict(value="projected_gcm"))
        return yhat

class ASDMultitask(DownscaleModel):
    def __init__(self, data, model, training_size=config.train_percent,
                 season=None, feature_columns=None,
                 ytransform=None, xtransform=None):
        DownscaleModel.__init__(self, data, model, training_size=training_size, season=season)
        self.xtransform = xtransform
        self.ytransform = ytransform

    def _split_dataset(self, X, y, test_set=False):
        if self.season:
            X = X[self.seasonidxs]
            y = y[self.seasonidxs]
        idx = int(X.shape[0] * self.training_size)
        if test_set:
            return X[idx:], y[idx:]
        else:
            return X[:idx], y[:idx]

    def train(self):
        y, self.locations = self.data.get_y()
        X = self.data.get_X()
        Xtrain, ytrain = self._split_dataset(X, y) 
        if self.xtransform is not None:
            self.xtransform.fit(Xtrain)
            Xtrain = self.xtransform.transform(Xtrain)
        if self.ytransform is not None:
            self.ytransform.fit(ytrain)
            ytrain = self.ytransform.transform(ytrain)
        self.model.fit(Xtrain, ytrain)

    def predict(self, test_set=True):
        y, self.locations = self.data.get_y()
        t = self.data.observations['time'].values
        X = self.data.get_X()
        t, _ = self._split_dataset(t, y, test_set=test_set)
        X, y = self._split_dataset(X, y) 
        if self.xtransform is not None:
            X = self.xtransform.transform(X)
        yhat = self.model.predict(X)
        if self.ytransform is not None:
            yhat = self.ytransform.inverse_transform(yhat)
        yhat = self.to_xray(yhat, t).rename({"value": "projected"})
        ytrue = self.to_xray(y, t).rename({"value": "ground_truth"})
        out = yhat.merge(ytrue)
        out['error'] = out.projected - out.ground_truth
        return out

    def to_xray(self, y, times):
        data = []
        for i, row in self.locations.iterrows():
            for j, t in enumerate(times):
                data.append({"value": y[j,i], self.data.obs_latdim: row[self.data.obs_latdim],
                             'time': t, self.data.obs_londim: row[self.data.obs_londim]})
        data = pandas.DataFrame(data)
        data.set_index([self.data.obs_latdim, self.data.obs_londim, "time"], inplace=True)
        data = xray.Dataset.from_dataframe(data)
        return data

    def project_gcm(self, gcm):
        X, t = gcm.get_X()
        nanrows = numpy.any(numpy.isnan(X), axis=1)
        X = X[~nanrows]
        t = t[~nanrows] 
        if self.xtransform is not None:
            X = self.xtransform.transform(X)
        yhat = self.model.predict(X)
        if self.ytransform is not None:
            yhat = self.ytransform.inverse_transform(yhat)
        yhat = self.to_xray(yhat, t) 
        yhat = yhat.rename({"value": "projected_gcm"})
        return yhat

