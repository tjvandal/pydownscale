import numpy
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import sys
from sklearn.linear_model.base import center_data

class BMA:
    def __init__(self, epochs=1000, eps=1e-5):
        self.epochs = epochs
        self.eps = eps

    def fit(self, X, y):
        self.K = X.shape[1]
        self.n = X.shape[0]
        self.a, self.b, self.f = self.get_coefficients(X, y)
        self.w = numpy.ones(self.K) / self.K
        Z = numpy.ones((self.n, self.K))
        sigma2 = self.sample_variance(Z, y)
        l = self.likelihood(y, sigma2)
        for j in range(self.epochs):
            Z = self.z_update(y, sigma2)
            self.w = Z.mean(axis=0)
            sigma2 = self.sample_variance(Z, y)
            if sum(numpy.isnan(sigma2)) > 0:
                z = Z.sum(axis=0)
                print z.shape
                print Z[:, z == 0]
                sys.exit()

            lold = l
            l = self.likelihood(y, sigma2)
            if numpy.abs(l - lold) < self.eps:
                break

        self.sigma2 = sigma2

    def sample_variance(self, Z, y):
        zsum = Z.sum(axis=0)
        zsum[zsum < numpy.finfo(float).eps] = 10*numpy.finfo(float).eps
        return numpy.sum((Z * (y[:, numpy.newaxis] - self.f))**2, axis=0) / zsum

    def z_update(self, y, sigma2):
        P = numpy.zeros(shape=(self.n, self.K))
        for k in range(self.K):
            P[:, k] = self.prob(y, k, sigma2)
        tot = P.sum(axis=1)
        Z = (P.T / tot).T
        return Z

    def get_coefficients(self, X, y):
        lm = LinearRegression(fit_intercept=True)
        a = numpy.zeros(self.K)
        b = numpy.zeros(self.K)
        f = numpy.zeros((self.n, self.K))
        for k in range(self.K):
            lm.fit(X, y)
            a[k] = lm.coef_[0]
            b[k] = lm.intercept_
            f[:, k] = a[k]*X[:, k] + b[k]

        return a, b, f

    def likelihood(self, y, sigma2):
        s = 0
        for k in range(self.K):
            s += self.w[k] * self.prob(y, k, sigma2)
        return numpy.log(s)

    def prob(self, y, k, sigma2):
        mu = self.f[:, k]
        sigma = numpy.sqrt(sigma2[k])
        if self.w[k] < 1e-10:   ## if w is low then the probability is irrelevant
            return 0
        return numpy.sum(norm.pdf(y, mu, sigma))

    def predict(self, X):
        #X = (X - self.X_mean) / self.X_std
        return (X * self.a + self.b).dot(self.w)



if __name__ == "__main__":
    from scipy.stats import pearsonr
    from sklearn import datasets


    data = datasets.load_boston()
    X = data["data"]
    y = data["target"]
    ntrain = 100

    idices = numpy.arange(X.shape[0])
    numpy.random.shuffle(idices)
    print idices

    model = BMA()
    model.fit(X[idices[:ntrain]], y[idices[:ntrain]])
    yhat = model.predict(X[idices[ntrain:]])
    print pearsonr(y[idices[ntrain:]], yhat)

    lm = LinearRegression(normalize=True)
    lm.fit(X[idices[:ntrain]], y[idices[:ntrain]])
    yhat = lm.predict(X[idices[ntrain:]])
    print pearsonr(y[idices[ntrain:]], yhat)

