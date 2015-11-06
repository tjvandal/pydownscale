__author__ = 'tj'
import sys
sys.path.insert(0, "/home/vandal.t/anaconda/lib/python2.7/site-packages")

import numpy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
from rMSSL import pMSSL
import time
from sklearn.utils.extmath import cartesian
import pandas

def test1_data():
    numpy.random.seed(0)
    t0 = time.time()
    n = 100
    d = 200
    k = 9
    W = numpy.random.normal(size=(d, k))
    rows1 = numpy.random.choice(range(d), 50)
    rows2 = numpy.random.choice(range(d), 50)

    W[rows1, :3] += numpy.random.normal(0, 2, size=(len(rows1), 1))
    W[rows2, 5:] += numpy.random.normal(0, 2, size=(len(rows2), 1))

    X = numpy.random.uniform(-1, 1, size=(n, d))
    X = X.dot(numpy.diag(1/numpy.sqrt(sum(X**2))))
    y = X.dot(W) + numpy.random.normal(0, 0.01, size=(n, k))
    return X, y, W

def test1_mpi():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    X, y, W = test1_data()
    ntrain = int(y.shape[0]*0.80)
    gspace = numpy.linspace(0, 4, 50)
    lspace = numpy.linspace(0, 2, 50)[1:]
    space = cartesian([gspace, lspace])

    if rank == 0:
        split = numpy.array_split(space, size)
    else:
        split = 0

    split = comm.scatter(split, root=0)
    results = []
    for g, l in split:
        try:
            mssl = pMSSL(max_epochs=1000, quite=True, gamma=g, lambd=l)
            mssl.fit(X[:ntrain], y[:ntrain], rho=1e-2)
            yhat = mssl.predict(X[ntrain:])
            mse = numpy.mean((yhat - y[ntrain:])**2)
            num_omega_zeros = numpy.sum(mssl.Omega == 0)
            num_w_zeros = numpy.sum(mssl.W == 0)
            d = {"gamma": g, "lambda": l, "mse": mse, "omega_zeros": num_omega_zeros, "w_zeros": num_w_zeros}
        except Exception as err:
            sys.stderr.write("Gamma: %f, Lambda: %f\n%s" % (g, l, err))
            d = {"error": 1, "gamma": g, "lambda": l}
        results.append(d)
        sys.stdout.write(str(d))

    newdata = comm.gather(results)
    if rank == 0:
        newdata = [item for l in newdata for item in l]   ## condense lists of lists
        data = pandas.DataFrame(newdata)
        data.to_csv("mssl-test1-mpi.csv", index=False)

def test1():
    X, y, W = test1_data()
    ntrain = int(y.shape[0]*0.80)
    g, l = 100, 100
    print "mssl"
    mssl = pMSSL(max_epochs=1000, quite=False, gamma=g, lambd=l)
    mssl.fit(X[:ntrain], y[:ntrain], rho=1e-2)
    yhat = mssl.predict(X[ntrain:])
    mse = numpy.mean((yhat - y[ntrain:])**2)
    num_omega_zeros = numpy.sum(mssl.Omega == 0)
    num_w_zeros = numpy.sum(mssl.W == 0)
    d = {"gamma": g, "lambda": l, "mse": mse, "omega_zeros": num_omega_zeros, "w_zeros": num_w_zeros}
    print d

if __name__ == "__main__":
    test1()