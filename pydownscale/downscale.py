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
    def __init__(self, data, model, max_train_year=config.train_percent, season=None, feature_columns=None):
        if data.__class__.__name__ != 'DownscaleData':
            raise TypeError("Data must be of type downscale data.")
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("Model must have methods train and predict.")

        self.data = data
        self.model = model
        self.season = season
        self.feature_columns = feature_columns
        self.max_train_year = max_train_year
        if self.season:
            key0 = self.data.reanalysis.keys()[0]
            self.seasonidxs = numpy.where(self.data.reanalysis[key0]['time.season'] == self.season)[0]

    # include quantile mapping ability
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

def asd_worker(model, X, y):
    model.fit(X, y)
    return model

def occurance_worker(model, X, y):
    return model.fit(X, y)

def worker_predict(model, X):
    return model.predict(X)

def worker_predict_prob(model, X):
    return model.predict_proba(X)[:,1]

def worker_invtrans(model, x):
    return model.inverse_transform(x)

class ASD(DownscaleModel):
    def __init__(self, data, model=BackwardStepwiseRegression(),
                 season=None, feature_columns=None, latdim='lat', londim='lon',
                 nearest_neighbor=True, ytransform=None, xtransform=None, num_proc=1, 
                max_train_year=config.max_train_year, conditional=None, cond_thres=None):
        DownscaleModel.__init__(self, data, model, max_train_year=max_train_year, season=season)
        self.nearest_neighbor = nearest_neighbor
        self.xtransform = xtransform
        self.ytransform = ytransform
        self.num_proc = num_proc
        self.conditional = conditional
        self.cond_thres = cond_thres

    def _split_dataset(self, X, test_set=False):
        if self.season:
            X = X[self.seasonidxs]
        idx = self.data.observations['time.year'][self.seasonidxs] <= self.max_train_year
        if test_set:
            return X[numpy.where(~idx)[0]]
        else:
            return X[numpy.where(idx)[0]]

    def train(self, location=None):
        y, self.locations = self.data.get_y(location=location)
        jobs = []
        cond_jobs = []
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
                Xtrain = self._split_dataset(X, test_set=False)
                if self.xtransform is not None:
                    self.xtrans += [copy.deepcopy(self.xtransform)]
                    self.xtrans[j].fit(Xtrain)
                    Xtrain = self.xtrans[j].transform(Xtrain)

            ytrain = self._split_dataset(y[:,j])
            ytrain = ytrain.reshape(-1,1)
            # Threshold before transformation
            if self.conditional is not None:
                cond_jobs += [delayed(occurance_worker)(copy.deepcopy(self.conditional),
                              Xtrain, ytrain.flatten() >= self.cond_thres)]
                occur_rows = ytrain.flatten() >= self.cond_thres
            else:
                occur_rows = [True] * ytrain.shape[0] 

            if self.ytransform is not None:
                self.ytrans += [copy.deepcopy(self.ytransform)]
                self.ytrans[j].fit(ytrain[occur_rows])
                ytrain = self.ytrans[j].transform(ytrain)
            jobs.append(delayed(asd_worker)(copy.deepcopy(self.model), Xtrain[occur_rows],
                                            ytrain.flatten()[occur_rows]))
        print "training models"
        self.models = Parallel(n_jobs=self.num_proc)(jobs)
        if self.conditional is not None:
            print "training occurance"
            self.occurance_models = Parallel(n_jobs=self.num_proc)(cond_jobs)

    def predict(self, test_set=True, location=None):
        Y, self.locations = self.data.get_y(location=location)
        t = self.data.observations['time'].values
        t = self._split_dataset(t, test_set=test_set)
        Y = self._split_dataset(Y, test_set=test_set)
        yhat_jobs = []
        ytrue =[]
        yoccur_jobs = []
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
            if self.conditional is not None:
                yoccur_jobs += [delayed(worker_predict_prob)(self.occurance_models[j], copy.deepcopy(X))]

            yhat_jobs += [delayed(worker_predict)(self.models[j], copy.deepcopy(X))]
            ytrue += [Y[:, j]]

        yhat = Parallel(n_jobs=self.num_proc)(yhat_jobs)
        if self.ytransform is not None:
            transform_jobs = [delayed(worker_invtrans)(self.ytrans[j], yhat[j]) for j in
                                                       range(len(yhat))]
            yhat = Parallel(n_jobs=self.num_proc)(transform_jobs)

        yhat = numpy.vstack(yhat).T
        ytrue = numpy.vstack(ytrue).T
        yhat = self.to_xray(yhat, t).rename({"value": "projected"})
        ytrue = self.to_xray(ytrue, t).rename({"value": "ground_truth"})
        if self.conditional is not None:
            yoccur = Parallel(n_jobs=self.num_proc)(yoccur_jobs)
            yoccur = numpy.vstack(yoccur).T > 0.5
            yoccur = self.to_xray(yoccur, t).rename({"value": "occurance"})
            yhat['projected'] = yhat['projected']*yoccur['occurance']
            yhat = yhat.merge(yoccur)

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
    def __init__(self, data, model, max_train_year=config.train_percent,
                 season=None, feature_columns=None, conditional=None,
                 ytransform=None, xtransform=None, cond_thres=None):
        DownscaleModel.__init__(self, data, model, max_train_year=max_train_year, season=season)
        self.xtransform = xtransform
        self.ytransform = ytransform
        self.conditional = conditional
        self.cond_thres = cond_thres

    def _split_dataset(self, X, y, test_set=False):
        year = self.data.observations['time.year']
        if self.season:
            X = X[self.seasonidxs]
            y = y[self.seasonidxs]
            year = year[self.seasonidxs]

        idx = (year <= self.max_train_year)
        if test_set:
            return X[numpy.where(~idx)[0]], y[numpy.where(~idx)[0]]
        else:
            return X[numpy.where(idx)[0]], y[numpy.where(idx)[0]]

    def train(self):
        y, self.locations = self.data.get_y()
        X = self.data.get_X()
        t = self.data.observations['time'].values
        Xtrain, ytrain = self._split_dataset(X, y, test_set=False) 
        yclassify = (ytrain >= self.cond_thres).copy() * 1.
        if self.xtransform is not None:
            self.xtransform.fit(Xtrain)
            Xtrain = self.xtransform.transform(Xtrain)

        if self.conditional is not None:
            occur_rows = (ytrain >= self.cond_thres).mean(axis=1) >= 0.10
            self.conditional.fit(Xtrain, yclassify)
        else:
            occur_rows = [True] * ytrain.shape[0]

        if self.ytransform is not None:
            self.ytransform.fit(ytrain)
            ytrain = self.ytransform.transform(ytrain)
        self.model.fit(Xtrain[occur_rows], ytrain[occur_rows])

    def predict(self, test_set=True):
        y, self.locations = self.data.get_y()
        t = self.data.observations['time'].values
        X = self.data.get_X()
        t, _ = self._split_dataset(t, y, test_set=test_set)
        X, y = self._split_dataset(X, y, test_set=test_set) 
        if self.xtransform is not None:
            X = self.xtransform.transform(X)
        yhat = self.model.predict(X)
        if self.ytransform is not None:
            yhat = self.ytransform.inverse_transform(yhat)
        yhat = self.to_xray(yhat, t).rename({"value": "projected"})
        ytrue = self.to_xray(y, t).rename({"value": "ground_truth"})
        if self.conditional is not None:
            yoccur = self.conditional.predict(X)
            yoccur = self.to_xray(yoccur, t).rename({"value": "occurance"})
            yhat['projected'] = yhat['projected']*(yoccur['occurance'] > 0.5)
            yhat = yhat.merge(yoccur)

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

if __name__ == "__main__": 
    import pickle
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.decomposition import PCA
    f = "/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/newengland_D_12781_8835.pkl"
    data = pickle.load(open(f, 'r'))
    asd = ASD(data, model=LinearRegression(),xtransform=PCA(n_components=0.98), season='JJA',
             conditional=LogisticRegression(), cond_thres=10., num_proc=40) 
    asd.train()
    print "predicting"
    yhat = asd.predict()
    print yhat
