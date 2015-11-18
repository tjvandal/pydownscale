__author__ = 'tj'
from sklearn.decomposition import PCA
from sklearn.svm import SVR
import numpy

class PCASVR:
    def __init__(self, explained_var=0.95):
        self.pca = PCA()
        self.svr = SVR()
        self.explained_var = explained_var

    def fit(self, X, y):
        X_t = self.pca.fit_transform(X)
        evr = numpy.cumsum(self.pca.explained_variance_ratio_)
        self.evr_idx = numpy.where(evr < self.explained_var)[0].max() + 1
        X_t = X_t[:,:(self.evr_idx+1)]
        print X.shape, X_t.shape, self.evr_idx
        self.svr.fit(X_t, y)

    def predict(self, X):
        X_t = self.pca.transform(X)
        X_t = X_t[:, :(self.evr_idx+1)]
        return self.svr.predict(X_t)

if __name__ == "__main__":
    from sklearn import datasets
    data = datasets.load_boston()
    X = data["data"]
    y = data["target"]
    model = PCASVR()
    model.fit(X, y)

