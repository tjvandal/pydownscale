from data import DownscaleData, read_nc_files
import numpy
from scipy.stats import pearsonr, spearmanr, kendalltau

class DownscaleModel:
    def __init__(self, data, model, training_size=0.70):
        if data.__class__.__name__ != 'DownscaleData':
            raise TypeError("Data must be of type downscale data.")
        if not hasattr(model, "fit") or not hasattr(model, "predict"):
            raise TypeError("Model must have methods train and predict.")

        self.data = data
        self.model = model
        self._split_dataset(training_size)

    def _split_dataset(self, training_size):
        X = self.data.get_X()
        self.numtrain = int(X.shape[0] * training_size)
        self.X_train = X[:self.numtrain]
        self.X_test = X[self.numtrain:]

    # include quantile mapping ability
    def train(self, location):
        self.location = location
        y = self.data.observations.loc[location].to_array().values.squeeze()
        self.y_train = y[:self.numtrain]
        self.y_test = y[self.numtrain:]
        self.model.fit(self.X_train, self.y_train)
        self.yhat_train = self.model.predict(self.X_train)
        self.yhat_test = self.model.predict(self.X_test)

    def get_mse(self, test_set=True):
        if test_set:
            return numpy.mean((self.y_test - self.yhat_test) ** 2)
        else:
            return numpy.mean((self.y_train - self.yhat_train) ** 2)

    # add functionality: plotting, quantile mapping, distirbutions, correlations
    def get_results(self, test_set=True):
        if test_set:
            y = self.y_test
            yhat = self.yhat_test
        else:
            y = self.y_train
            yhat = self.yhat_train
        results = self.location   ## dictionary of lat, lon
        results['rmse'] = self.get_mse(test_set)**(0.5)
        results['pearson'] = pearsonr(y, yhat)[0]
        results['spearman'] = spearmanr(y, yhat)[0]
        results['kendaltau'] = kendalltau(y, yhat)[0]
        results['yhat_mean'] = numpy.mean(yhat)
        results['y_mean'] = numpy.mean(y)
        results['yhat_std'] = numpy.std(yhat)
        results['y_std'] = numpy.std(y)
        return results


if __name__ == "__main__":
    import pandas
    from sklearn.linear_model import LassoCV
    cmip5_dir = "/Users/tj/data/cmip5/access1-0/"
    cpc_dir = "/Users/tj/data/usa_cpc_nc/merged"

    # climate model data, monthly
    cmip5 = read_nc_files(cmip5_dir)
    cmip5 = cmip5.resample('MS', 'time', how='mean')   ## try to not resample

    # daily data to monthly
    cpc = read_nc_files(cpc_dir)
    monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

    data = DownscaleData(cmip5, monthlycpc)
    linearmodel = LassoCV(alphas=numpy.array([0.1, 1, 10]))
    dmodel = DownscaleModel(data, linearmodel)
    dmodel.train(location={'lat': 31.875, 'lon': -81.375})
    print dmodel.get_results()