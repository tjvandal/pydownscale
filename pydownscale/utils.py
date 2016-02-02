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

if __name__ == "__main__":
    import numpy
    x = numpy.random.uniform(size = (10,100))
    y, l, ymean, ystd = center_boxcox(x)
    print boxcox(x[:,1], 4.42169476)
