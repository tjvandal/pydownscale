from data import DownscaleData, read_nc_files
import numpy

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

if __name__ == "__main__":
    import pandas
    from sklearn.linear_model import LassoCV
    cmip5_dir = "/Users/tj/data/cmip5/access1-0/"
    cpc_dir = "/Users/tj/data/usa_cpc_nc/merged"

    # climate model data, monthly
    cmip5 = read_nc_files(cmip5_dir)
    cmip5.time = pandas.to_datetime(cmip5.  time.values)
    cmip5 = cmip5.resample('MS', 'time', how='mean')   ## try to not resample

    # daily data to monthly
    cpc = read_nc_files(cpc_dir)
    cpc.time = pandas.to_datetime(cpc.time.values)
    monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

    data = DownscaleData(cmip5, monthlycpc)
    linearmodel = LassoCV(alphas=numpy.array([0.1, 1, 10]))
    dmodel = DownscaleModel(data, linearmodel)
    dmodel.train(location={'lat': 31.875, 'lon': -81.375})
    print dmodel.get_mse(test_set=False), dmodel.get_mse()