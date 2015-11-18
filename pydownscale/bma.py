import numpy
from sklearn.linear_model import LinearRegression
from scipy.stats import norm
import sys

class BMA:
    def __init__(self):
        pass

    def fit(self, X, y, epochs=1000, eps=1e-5):
        self.K = X.shape[1]
        self.n = X.shape[0]
        self.X = X
        self.y = y
        self.a, self.b, self.f = self.get_coefficients()
        self.w = numpy.ones(self.K) / self.K
        Z = numpy.ones((self.n, self.K))
        sigma2 = self.sample_variance(Z)
        l = self.likelihood(sigma2)
        for j in range(epochs):
            Z = self.z_update(sigma2)
            self.w = Z.mean(axis=0)
            sigma2 = self.sample_variance(Z)
            if sum(numpy.isnan(sigma2)) > 0:
                print Z.shape
                z = Z.sum(axis=0)
                print z.shape
                print Z[:, z == 0]
                sys.exit()

            lold = l
            l = self.likelihood(sigma2)
            if numpy.abs(l - lold) < eps:
                break

        self.sigma2 = sigma2


    def sample_variance(self, Z):
        zsum = Z.sum(axis=0)
        zsum[zsum < numpy.finfo(float).eps] = 10*numpy.finfo(float).eps
        return numpy.sum((Z * (self.y[:, numpy.newaxis] - self.f))**2, axis=0) / zsum

    def z_update(self, sigma2):
        P = numpy.zeros(shape=(self.n, self.K))
        for k in range(self.K):
            P[:, k] = self.prob(k, sigma2)
        tot = P.sum(axis=1)
        Z = (P.T / tot).T
        return Z

    def get_coefficients(self):
        lm = LinearRegression(fit_intercept=True)
        a = numpy.zeros(self.K)
        b = numpy.zeros(self.K)
        f = numpy.zeros((self.n, self.K))
        for k in range(self.K):
            lm.fit(self.X[:, k][:, numpy.newaxis], self.y)
            a[k] = lm.coef_[0]
            b[k] = lm.intercept_
            f[:, k] = a[k]*self.X[:, k] + b[k]

        return a, b, f

    def likelihood(self, sigma2):
        s = 0
        for k in range(self.K):
            s += self.w[k] * self.prob(k, sigma2)
        return numpy.log(s)

    def prob(self, k, sigma2):
        mu = self.f[:, k]
        sigma = numpy.sqrt(sigma2[k])
        if self.w[k] < 1e-10:   ## if w is low then the probability is irrelevant
            return 0
        return numpy.sum(norm.pdf(self.y, mu, sigma))

    def predict(self, X):
        return (X * self.a + self.b).dot(self.w)



if __name__ == "__main__":
    from scipy.stats import pearsonr
    from sklearn import datasets
    data = datasets.load_boston()
    X = data["data"]
    y = data["target"]
    ntrain = 30

    model = BMA()
    model.fit(X[:ntrain], y[:ntrain])
    yhat = model.predict(X[ntrain:])
    print pearsonr(y[ntrain:], yhat)

    lm = LinearRegression()
    lm.fit(X[:ntrain], y[:ntrain])
    yhat = lm.predict(X[ntrain:])
    print pearsonr(y[ntrain:], yhat)

