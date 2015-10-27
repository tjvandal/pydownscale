__author__ = 'tj'
import numpy
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot
from data import DownscaleData, read_nc_files
from downscale import DownscaleModel
import sys
from scipy.linalg import solve_sylvester
import time

class pMSSL:
    def __init__(self, max_epochs=1000, quite=False):
        self.max_epochs = max_epochs
        self.quite = quite

    def train(self,X, y, lambd=1e-4, gamma=1e-4, rho=1e-4, wadmm=True):
        self.X = X
        self.y = y
        self.rho = rho
        self.gamma = gamma
        self.K = self.y.shape[1]
        self.n = self.y.shape[0]
        self.d = self.X.shape[1]
        self.Omega = numpy.eye(self.K)
        self.W = numpy.zeros(shape=(self.d, self.K))
        self.lambd = lambd
        print "Number of tasks: %i, Number of dimensions: %i, Number of observations: %i, Lambda: %0.2f, Gamma: %0.2f" % (self.K, self.d, self.n, self.lambd, self.gamma)
        costdiff = 10
        omegatime = 0.
        wtime = 0.
        t = 0
        costs = []
        while (costdiff > 10e-3) and (t < self.max_epochs):
            tw = time.time()
            if wadmm:
                self.W = self._w_update_admm(rho=self.rho)
            else:
                self.W = self._w_update()
            wtime += (time.time() - tw)
            to = time.time()
            self.Omega = self._omega_update(self.Omega, rho=self.rho)
            omegatime += (time.time() - to)
            curr_cost = self.cost(self.W, self.Omega)
            costs.append(curr_cost)
            if t == 0:
                costdiff = curr_cost
                prevcost = curr_cost
            else:
                costdiff = numpy.abs(prevcost - curr_cost)
                prevcost = curr_cost
            t += 1
            if not self.quite:
                print "iteration %i, costdiff: %f" % (t, costdiff)  

        print "Amount of time to train Omega: %f" % omegatime
        print "Amount of time to train W:     %f" % wtime

    def cost(self, W, Omega):
        cost, _ = self._w_cost(W)
        cost += -self.K/2.*numpy.log(numpy.linalg.det(Omega)) 
        cost += self.lambd * numpy.linalg.norm(Omega, 1)
        return cost 

    def shrinkage_threshold(self, a, alpha):
        return numpy.maximum(numpy.zeros(shape=a.shape), a-alpha) - numpy.maximum(numpy.zeros(shape=a.shape), -a-alpha)

    def _softthres(self, x, thres):
        if x > thres:
            return x - thres
        elif numpy.abs(x) < thres:
            return 0
        else:
            return x+thres

    def softthreshold(self, X, thres):
        return numpy.piecewise(X, 
            [X > thres, numpy.abs(X) <= thres, X < -thres], 
            [lambda X: X - thres, 0, lambda X: X+thres])

    def _omega_update(self, Omega, rho):
        maxrho = 10
        Z = numpy.zeros(shape=(self.K, self.K))
        U = numpy.zeros(shape=(self.K, self.K))
        j = 0
        dualresid = 10e6
        resid = []
        S = self.W.T.dot(self.W)
        epsabs = 1e-3
        epsrel = 1e-3
        epsdual = numpy.sqrt(self.n) * epsabs + epsrel * numpy.linalg.norm(self.y,2)
        for j in range(1000): # force 10 iterations,
            L, Q = numpy.linalg.eig(self.rho * (Z - U) - S)
            Omega_tilde = numpy.eye(self.K)
            numpy.fill_diagonal(Omega_tilde, (L + numpy.sqrt(L**2 + 4*rho))/(2*rho))
            Omega = Q.dot(Omega_tilde).dot(Q.T)
            Z_prev = Z.copy()
            Z = self.softthreshold(Omega + U, self.lambd/rho)
            U = U + Omega - Z
            
            dualresid = numpy.linalg.norm(self.rho * (Z - Z_prev), 2)
            primalresid = numpy.linalg.norm(Omega - Z, 2)
            epspri = numpy.sqrt(self.d) * epsabs + epsrel * numpy.max([numpy.linalg.norm(Omega, 2), numpy.linalg.norm(Z, 2), 0])

            rho = min(rho*1.1, maxrho)
            if (j % 500) == 1 and (not self.quite):
                print "omega update:", j, "Dualresid:", dualresid

            if (dualresid < epsdual) and (primalresid < epspri):
                break


        #if not self.quite:
        return Omega

   # Lets save the proximal descent update
    def _w_update(self):
        costdiff = 10e6
        W = self.W
        j = 0
        t0 = time.time()
        tk = 1/(2*numpy.linalg.norm(self.X.T.dot(self.X), 1)) # tk exits in (0, 1/||X.T*X||)
        XX = self.X.T.dot(self.X)
        XY = self.X.T.dot(self.y)
        while costdiff > 1e-6:
            cost, gmat = self._w_cost(W, XX=XX, XY=XY)
            W = self.shrinkage_threshold(W - tk*gmat, alpha=self.lambd*tk)
            if j == 0:
                costdiff = numpy.abs(cost)
            else:
                costdiff = numpy.abs(costprev - cost)
            costprev = cost
            if (j > 10000):
                print "Warning: W did not converge."
                break
            j += 1

        return W

    # Lets parallelize this
    def _w_cost(self, W, XX=None, XY=None):
        XW = self.X.dot(W)
        if XX is None:
            XX = self.X.T.dot(self.X)
        if XY is None:
            XY = self.X.T.dot(self.y)
        f = (self.y-XW).T.dot((self.y-XW)) / (2*len(self.y))
        f += self.lambd*numpy.trace(W.dot(self.Omega).dot(W.T))
        gmat = (XX.dot(W) - XY)/len(self.y)  # the gradients
        gmat += 2*W.dot(self.Omega) # *self.lambd
        return numpy.sum(f), gmat

    def _w_update_admm(self, rho):
        maxrho = 10
        Z = numpy.zeros(shape=(self.d, self.K))
        U = numpy.zeros(shape=(self.d, self.K))
        j = 0
        XX = self.X.T.dot(self.X)  # dxd
        Xy = self.X.T.dot(self.y)  # dxk
        Theta = self.W.copy()      # dxk
        epsabs = 1e-3
        epsrel = 1e-3
        epsdual = numpy.sqrt(self.n) * epsabs + epsrel * numpy.linalg.norm(self.y,2)
        tsum = 0.
        for j in range(500):
            prevTheta = Theta.copy()
            C = Xy + rho * (Z - U)
            t0 = time.time()
            Theta = solve_sylvester(XX + rho * numpy.eye(XX.shape[0]), self.Omega, C)
            tsum += time.time() - t0
            Z_prev = Z.copy()
            Z = self.softthreshold(Theta + U, self.gamma/rho)
            U = U + Theta - Z
            if (j % 100 == 0) and (not self.quite):
                print j, numpy.linalg.norm(prevTheta - Theta, 2)
            rho = min(rho*1.1, maxrho)

            dualresid = numpy.linalg.norm(self.rho*(Z - Z_prev), 2)
            primalresid = numpy.linalg.norm(Theta - Z, 2)
            epspri = numpy.sqrt(self.d) * epsabs + epsrel * numpy.max([numpy.linalg.norm(Theta, 2), numpy.linalg.norm(Z, 2), 0])
            
            if (dualresid < epsdual) and (primalresid < epspri):
                break
            
        print "Time to solve W with ADMM:", tsum, j
        return Theta

    def predict(self, X):
        return X.dot(self.W)

def test1():
    import time
    t0 = time.time()
    n = 10000
    ntrain = int(n*0.80)
    d = 100
    k = 9
    W = numpy.random.normal(size=(d, k))
    W[:, :20] += numpy.random.normal(0, 10, size=(d, 1))
    W[:, 20:100] += numpy.random.normal(0, 10, size=(d, 1))
    print W[:5,:5]
    X = numpy.random.uniform(-1, 1, size=(n, d))
    y = X.dot(W)
    mssl = pMSSL(max_epochs=200, quite=True)
    MSE = numpy.zeros((5,5))
    for j, g in enumerate(10**numpy.linspace(-1, 2, 4)):
        for i, l in enumerate(10**numpy.linspace(-1, 2, 4)):
            try:
                mssl.train(X[:70], y[:70], rho=1e-4, gamma=g, lambd=l)
            except:
                print "Pass -- Lamdba %f, Gamma %f" % (l, g)
                continue
            yhat = mssl.predict(X[70:])
            mse = numpy.mean((yhat - y[70:])**2)
            MSE[i, j] = mse
            print mse
            try:
                if mse < bestmse:
                    bestmssl = mssl
            except NameError:
                bestmssl = mssl
    print MSE
    pyplot.subplot(2,1,1)
    pyplot.imshow(MSE, interpolation="none")
    pyplot.subplot(2,1,2)
    pyplot.imshow(bestmssl.Omega, interpolation="none", cmap="Reds")
    pyplot.savefig("test2fig.pdf")
    print "Time to run:", time.time() - t0


def test2():
    n = 10000
    d = 1000
    k = 1000
    l = 1e1
    g = 1e1
    tt = time.time()
    W = numpy.random.normal(size=(d, k))
    W[:, :4] += numpy.random.normal(0, 10, size=(d, 1))
    W[:, 5:10] += numpy.random.normal(0, 10, size=(d, 1))
    #print W[:5,:15]
    X = numpy.random.uniform(-1, 1, size=(n, d))
    y = X.dot(W)
    mssl = pMSSL(max_epochs=50, quite=True)
    mssl.train(X, y, rho=1e-4, gamma=g, lambd=l, wadmm=False)
    mssl.Omega[numpy.abs(mssl.Omega) < 1e-10] = 0
    #print mssl.W[:5, :15]
    #print mssl.Omega
    #pyplot.imshow(numpy.linalg.inv(mssl.Omega))
    pyplot.imshow(mssl.Omega, interpolation="none", cmap="Reds")
    pyplot.title("Lamdba %f, Gamma %f" % (l, g))
    pyplot.savefig("test1.pdf")
    print "Time to train: ", time.time() - tt
    return mssl

def climatetest():
    import time
    t0 = time.time()
    cmip5_dir = "/scratch/vandal.t/cmip5/access1-3/"
    cpc_dir = "/scratch/vandal.t/merged/"

    # climate model data, monthly
    cmip5 = read_nc_files(cmip5_dir)
    cmip5.load()
    cmip5 = cmip5.resample('MS', 'time', how='mean')   ## try to not resample

    # daily data to monthly
    cpc = read_nc_files(cpc_dir)
    cpc.load()
    monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

    print "Data Loaded: %d seconds" % (time.time() - t0)
    data = DownscaleData(cmip5, monthlycpc)
    #data.normalize_monthly()

    # print "Data Normalized: %d" % (time.time() - t0)
    X = data.get_X()
    lats = data.observations.lat
    lons = data.observations.lon
    j = len(lats)/2
    i = len(lons)/2
    Y = data.observations.loc[{'lat': lats[j:j+5], 'lon': lons[i:i+5]}].to_array().values.squeeze()
    Y = Y.reshape(Y.shape[0], Y.shape[1]*Y.shape[2])
    
    mssl = pMSSL()
    mssl.train(X[:70,:5000], Y[:70])
    yhat = mssl.predict(X[70:,:5000])
    #pyplot.plot(yhat[70:,0])
    #pyplot.plot(Y[70:, 0], color='red')
    #pyplot.show()

if __name__ == "__main__":
    ms = test2()
    #climatetest()

