import numpy
import sys
from scipy.stats import boxcox
from scipy.special import inv_boxcox

def center_frob(x):
    x_mean = x.mean(axis=0)
    x_frob = numpy.diag(1./numpy.sqrt(numpy.sum(x**2, axis=0)))
    x = x.dot(x_frob)
    return x, x_frob

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

class BoxcoxTransform():
    def __init__(self, minlmbda=-2, maxlmbda=1): 
        self.minlmbda = minlmbda
        self.maxlmbda = maxlmbda

    def fit(self, X):
        xtrans = numpy.zeros(shape=X.shape)
        if len(X.shape) == 2:
            self.shift = -X.min(axis=0)
            self.shift[self.shift < 0] = 0
            self.shift += 3 * X.std(axis=0)
            X += self.shift
            self.lmbda = numpy.zeros(X.shape[1])
            for j in range(X.shape[1]):
                _, self.lmbda[j]= boxcox(X[:, j])
                self.lmbda[j] = max(self.lmbda[j], self.minlmbda)
                self.lmbda[j] = min(self.lmbda[j], self.maxlmbda)
                if numpy.abs(self.lmbda[j]) < 1e-4:
                    self.lmbda[j] = 0
                    print "changing lambda"
                xtrans[:, j] = boxcox(X[:, j], self.lmbda[j])
        elif len(X.shape) == 1:
            self.shift = max([1e-10,-X.min()])
            self.shift += 3 * X.std()
            X += self.shift 
            xtrans, self.lmbda = boxcox(X)
        self.xmean = xtrans.mean(axis=0)
        self.xstd = xtrans.std(axis=0)

    def transform(self, X):
        Xcopy = X.copy()
        X += self.shift
        if isinstance(self.lmbda, float):
            xb = boxcox(X, self.lmbda)
        else:
            xb = numpy.zeros(shape=X.shape)
            for j, lmb in enumerate(self.lmbda):
                xb[:, j] = boxcox(X[:, j], lmb)
        return (xb - self.xmean) / self.xstd 

    def inverse_transform(self, X):
        xtemp = X*self.xstd + self.xmean
        if isinstance(self.lmbda, float):
            xinv = inv_boxcox(xtemp, self.lmbda)
        else:
            xinv = numpy.zeros(shape=X.shape)
            for j, lmb in enumerate(self.lmbda):
                xinv[:, j] = inv_boxcox(xtemp[:, j], lmb)
        return xinv - self.shift 

if __name__ == "__main__":
    from matplotlib import pyplot
    import numpy
    x = numpy.random.lognormal(0, 1, size = (10,100))
    trans = BoxcoxTransform()
    trans.fit(x)
    xtran = trans.transform(x)
    xinvtran = trans.inverse_transform(xtran)
    pyplot.subplot(2,1,1)
    pyplot.hist(x.flatten())
    pyplot.subplot(2,1,2)
    pyplot.hist(xtran.flatten())
    pyplot.show()
