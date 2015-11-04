__author__ = 'tj'
import sys
sys.path.insert(0, "/home/vandal.t/anaconda/lib/python2.7/site-packages")

from mpi4py import MPI
import numpy
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot
from rMSSL import pMSSL
import time
from sklearn.utils.extmath import cartesian
import pandas

def test1():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    t0 = time.time()
    n = 100
    ntrain = int(n*0.80)
    d = 200
    k = 9
    W = numpy.random.normal(size=(d, k))
    rows1 = numpy.random.choice(range(d), 5)
    rows2 = numpy.random.choice(range(d), 6)

    W[rows1, :3] += numpy.random.normal(0, 2, size=(len(rows1), 1))
    W[rows2, 5:] += numpy.random.normal(0, 2, size=(len(rows2), 1))

    X = numpy.random.uniform(-1, 1, size=(n, d))
    X = X.dot(numpy.diag(1/numpy.sqrt(sum(X**2))))
    y = X.dot(W) + numpy.random.normal(0, 0.01, size=(n, k))

    nl = 3
    gl = 3


    gspace = numpy.linspace(0, 10, 50)
    lspace = numpy.linspace(0, 10, 50)
    space = cartesian([gspace, lspace])

    if rank == 0:
        split = numpy.split(space, size)
    else:
        split = 0

    split = comm.scatter(split, root=0)
    results = []
    for g, l in split:
        mssl = pMSSL(max_epochs=200, quite=True, gamma=g, lambd=l)
        mssl.fit(X[:ntrain], y[:ntrain], rho=1e-2)
        yhat = mssl.predict(X[ntrain:])
        mse = numpy.mean((yhat - y[ntrain:])**2)
        num_omega_zeros = numpy.sum(mssl.Omega == 0)
        num_w_zeros = numpy.sum(mssl.W == 0)
        d = {"gamma": g, "lambda": l, "mse": mse, "omega_zeros": num_omega_zeros, "w_zeros": num_w_zeros}
        results.append(d)

    newdata = comm.gather(results)
    if rank == 0:
        newdata = [item for l in newdata for item in l]   ## condense lists of lists
        data = pandas.DataFrame(newdata)
        data.to_csv("mssl-test1-mpi.csv", index=False)

if __name__ == "__main__":
    test1()