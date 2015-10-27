#!/shared/apps/sage/sage-5.12/spkg/bin/sage -python

'''
What to do
Test scalability of the algorithm using different values of d and k
for k in [1,10,100,1000]
	for d in [10,100,1000]:

'''
import sys
sys.path.insert(0, "/home/vandal.t/anaconda/lib/python2.7/site-packages")

from mpi4py import MPI
from pydownscale.rMSSL import pMSSL
import time
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
import pandas

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

tasks = [10, 100, 1000]
dims = [10, 100, 1000]
admmw = [True, False]

n = 100
ntrain = int(0.80*n)
l = 1e-3
g = 1e-3

if rank==0:
	combs = numpy.array([(t, d, a) for t in tasks for d in dims for a in admmw])
	combs = numpy.split(combs, size)
else:
	combs = None

combs = comm.scatter(combs, root=0)
print combs, rank
results = []
for c in combs:
	numpy.random.seed(1)
	k = c[0]
	d = c[1]
	adw = c[2]

	print "Tasks: %i, Covariates: %i" % (k, d)
	W = numpy.random.normal(size=(d, k))
	W[:, :4] += numpy.random.normal(0, 10, size=(d, 1))
	W[:, 5:10] += numpy.random.normal(0, 10, size=(d, 1))
	X = numpy.random.uniform(-1, 1, size=(n, d))
	y = X.dot(W)
	res = {"tasks": k, "covariates": d, "lambda": l, "gamma": g, 
			"n": n}
	t0 = time.time()
	mssl = pMSSL(max_epochs=100, quite=True)
	mssl.train(X[:ntrain], y[:ntrain], rho=1e-4, gamma=g, lambd=l, wadmm=adw)
	yhat = mssl.predict(X[ntrain:])
	res["mse"] = numpy.mean((yhat - y[ntrain:])**2)
	res['time'] = time.time() - t0
	res["admm"] = adw

	results.append(res)
	sys.stdout.write(str(res))

newData = comm.gather(results, root=0)
if rank == 0:
   #print newData
   newData = [item for l in newData for item in l]   ## condense lists of lists
   print newData
   data = pandas.DataFrame(newData)
   data.to_csv("admm-results-9.csv", index=False)
   print data