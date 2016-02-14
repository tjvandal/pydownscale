import numpy
import sys
from scipy.stats import boxcox

def center_frob(x):
    x_mean = x.mean(axis=0)
    x_frob = numpy.diag(1./numpy.sqrt(numpy.sum(x**2, axis=0)))
    x = x.dot(x_frob)
    return x, x_frob

def center_log(x, axis=0):
    x = numpy.log(x + 1e-10)
    x_mean = x.mean(axis=axis)
    x_std = x.std(axis=axis)
    x = (x - x_mean) / x_std
    return x, x_mean, x_std

def center_boxcox(x, shift=0):
    x += shift
    if len(x.shape) > 1:
        lmbda = numpy.zeros(x.shape[1])
        for j in range(x.shape[1]):
            x[:,j], lmbda[j] = boxcox(x[:,j]+1e-10)
            if lmbda[j] < 0:
                lmbda[j] = 0
                x[:,j], _  = boxcox(x[:,j]+1e-10, lmbda[j])
    elif len(x.shape) == 1:
        x, lmbda = boxcox(x + 1e-10)
    xmean = x.mean(axis=0)
    xstd = x.std(axis=0)
    x = (x - xmean) / xstd
    return x, lmbda, xmean, xstd

class LogTransform():
    def __init__(self):
        pass

    def fit(self,X):
        if numpy.any(X < 0):
            raise ValueError("Log Transform: All values must be greater than or equal to zero")
        xlog = numpy.log(X+1e-10)
        self.xmean = xlog.mean(axis=0)
        self.xstd = xlog.std(axis=0)

    def transform(self,X):
        xlog = numpy.log(X+1e-10)
        return (xlog - self.xmean)/self.xstd

    def inverse_transform(self,X):
        xinv = X*self.xstd + self.xmean
        xinv = numpy.exp(xinv)-1e-10
        return xinv

if __name__ == "__main__":
    import numpy
    x = numpy.random.lognormal(0, 1, size = (10,100))
    trans = LogTransform()
    trans.fit(x)
    xtran = trans.transform(x)
    xinvtran = trans.inverse_transform(xtran)

