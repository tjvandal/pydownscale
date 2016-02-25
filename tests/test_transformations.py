import sys
sys.path.append("/home/vandal.t/repos/pydownscale/pydownscale")
import numpy
import scipy.stats
from utils import LogTransform, BoxcoxTransform

def test_log():
    n, k = 100, 10
    y = numpy.random.lognormal(size=(n,k))
    logtrans = LogTransform()
    logtrans.fit(y)
    ytran = logtrans.transform(y)
    ytraninv = logtrans.inverse_transform(ytran)
    print numpy.histogram((ytraninv - y).flatten())

def test_boxcox():
    n, k = 100, 10
    y = numpy.zeros(shape=(n,k))
    y[:,:2] = numpy.random.lognormal(size=(n,2))
    y[:, 2:5] = numpy.random.normal(size=(n,3))
    y[:, 5:8] = numpy.random.exponential(size=(n,3))
    y[:, 8:] = numpy.random.beta(1,1, size=(n,2))
    trans = BoxcoxTransform(minlmbda=-1, maxlmbda=2)
    trans.fit(y)
    ytrans = trans.transform(y)
    ytransinv = trans.inverse_transform(ytrans)
    print numpy.histogram((ytransinv - y).flatten())

if __name__ == "__main__":
    #test_log()
    test_boxcox()
