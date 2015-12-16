#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python
'''
local testing command:
mpirun -np 1 /shared/apps/sage/sage-5.12/spkg/bin/sage -python mssl_tests.py
'''

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
import pickle
from data import DownscaleData
from downscale import DownscaleModel

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
    X = X.dot(numpy.diag(1./numpy.sqrt(numpy.sum(X**2, axis=0))))
    y = X.dot(W) + numpy.random.normal(0, 0.01, size=(n, k))
    return X, y, W

def test1_mpi():
    from mpi4py import MPI

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    
    X, y, W = test1_data()
    ntrain = int(y.shape[0]*0.80)
    gspace = numpy.linspace(1e-4, 4, 10)
    lspace = numpy.linspace(1e-4, 2, 10)

    if rank == 0:
        space = numpy.asarray([[g, l] for g in gspace for l in lspace])
        split = numpy.array_split(space, size)

    else:
        split = None

    split = comm.scatter(split, root=0)
    results = []
    for g, l in split:
        try:
            mssl = pMSSL(max_epochs=1000, quite=True, gamma=g, lambd=l)
            mssl.fit(X[:ntrain], y[:ntrain])
            yhat = mssl.predict(X[ntrain:])
            mse = numpy.mean((yhat - y[ntrain:])**2)
            num_omega_zeros = numpy.sum(mssl.Omega == 0)
            num_w_zeros = numpy.sum(mssl.W == 0)
            d = {"gamma": g, "lambda": l, "mse": mse, 
            "omega_zeros": num_omega_zeros, "w_zeros": num_w_zeros}
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
    print "X shape", X.shape, " Y shape:", y.shape
    ntrain = int(y.shape[0]*0.80)
    g, l = 1e-3, 1e-4
    print "mssl"
    mssl = pMSSL(max_epochs=1000, quite=True, gamma=g, lambd=l)
    mssl.fit(X[:ntrain], y[:ntrain])
    yhat = mssl.predict(X[ntrain:])
    mse = numpy.mean((yhat - y[ntrain:])**2)
    num_omega_zeros = numpy.sum(mssl.Omega == 0)
    num_w_zeros = numpy.sum(mssl.W == 0)
    d = {"gamma": g, "lambda": l, "mse": mse, "omega_zeros": num_omega_zeros, "w_zeros": num_w_zeros}
    #print mssl.W
    print d

def climate():

    lambd = 1e2
    gamma = 1e2
    season = "DJF"
    data = pickle.load(open("/scratch/vandal.t/experiments/DownscaleData/monthly_804_3150.pkl", "r"))
    model = pMSSL(lambd=lambd, gamma=gamma, max_epochs=1000)
    dmodel = DownscaleModel(data, model, season=season)
    dmodel.train()
    print dmodel.model.Omega[:5, :5]
    print dmodel.model.W[:5, :5]
    print "Omega to zero", numpy.sum(dmodel.model.Omega == 0)
    print "W to zero", numpy.sum(dmodel.model.W == 0)

if __name__ == "__main__":
    #climate()
    #test1()
    test1_mpi()

