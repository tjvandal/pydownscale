__author__ = 'tj'
import numpy
from matplotlib import pyplot
from data import DownscaleData, read_nc_files
from downscale import DownscaleModel
import sys
from scipy.linalg import solve_sylvester

numpy.random.seed(1)

class pMSSL:
    def __init__(self, max_epochs=1000, quite=False):
        self.max_epochs = max_epochs
        self.quite = quite

    def train(self,X, y, lambd=0.0001, gamma=0.0001, rho=.1):
        self.X = X
        self.y = y
        self.rho = rho
        self.gamma = gamma
        self.K = self.y.shape[1]
        self.n = self.y.shape[0]
        self.d = self.X.shape[1]
        print "Number of tasks: %i, Number of dimensions: %i, Number of observations: %i" % (self.K, self.d, self.n)
        self.Omega = numpy.eye(self.K)
        self.W = numpy.zeros(shape=(self.d, self.K))
        self.lambd = lambd
        costdiff = 10
        t = 0
        costs = []
        while (costdiff > 10e-6) and (t < self.max_epochs):
            self.W = self._w_update_admm(rho=self.rho)
            #print self.W[:5, :5]
            #self.W = self._w_update()
            #print self.W[:5,:5]
            self.Omega = self._omega_update(self.Omega, rho=self.rho)
            curr_cost, _ = self._w_cost(self.W)
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
        while (dualresid > 1e-6) or (j < 10): # force 10 iterations,
            L, Q = numpy.linalg.eig(self.rho * (Z - U) - S)
            Omega_tilde = numpy.eye(self.K)
            numpy.fill_diagonal(Omega_tilde, (L + numpy.sqrt(L**2 + 4*rho))/(2*rho))
            Omega = Q.dot(Omega_tilde).dot(Q.T)
            Z_prev = Z.copy()
            Z = self.softthreshold(Omega + U, self.lambd/rho)
            U = U + Omega - Z
            dualresid = numpy.linalg.norm(self.rho * self.X.T.dot(self.y).dot(Z - Z_prev), 2)
           # dualresid = numpy.linalg.norm(self.rho * (Z - Z_prev), 2)
            Z = self.softthreshold(Omega + U, self.lambd/rho)
            rho = min(rho*1.1, maxrho)
            if (j % 500) == 1 and (not self.quite):
                print "omega update:", j, "Dualresid:", dualresid
            j+=1
            resid.append(dualresid)
            if j % 100 == 0:
                #pyplot.plot(resid)
                #pyplot.draw()
                #pyplot.show()
                resid = []
        if not self.quite:
            print "Omega Converged at", j
        return Omega

    def _w_update(self):
        costdiff = 10e6
        W = self.W
        j = 0
        tk = 1/(2*numpy.linalg.norm(self.X.T.dot(self.X), 1)) # tk exits in (0, 1/||X.T*X||)
        while costdiff > 0.01:
            cost, gmat = self._w_cost(W)
            W = self.shrinkage_threshold(W - tk*gmat, alpha=self.lambd*tk)

            if j == 0:
                costdiff = numpy.abs(cost)
            else:
                costdiff = numpy.abs(costprev - cost)
            costprev = cost
            if (j > 10000) and (not self.quite):
                print "Warning: W did not converge."
                break
            j += 1
        return W


    # Lets parallelize this
    def _w_cost(self, W):
        XW = self.X.dot(W)
        f = (self.y-XW).T.dot((self.y-XW)) / (2*len(self.y))
        f += self.lambd*numpy.trace(W.dot(self.Omega).dot(W.T))
        gmat = (self.X.T.dot(XW) - self.X.T.dot(self.y))/len(self.y)  # the gradients
        gmat += 2*W.dot(self.Omega) # *self.lambd
        return numpy.sum(f), gmat

    def _w_update_admm(self, rho):
        maxrho = 5
        Z = numpy.zeros(shape=(self.d, self.K))
        U = numpy.zeros(shape=(self.d, self.K))
        j = 0
        XX = self.X.T.dot(self.X)
        Xy = self.X.T.dot(self.y)
        Theta = self.W.copy()
        for j in range(500):
            prevTheta = Theta.copy()
            C = Xy + rho * (Z - U)
            Theta = solve_sylvester(XX + rho * numpy.eye(XX.shape[0]), self.Omega, C)
            Z = self.softthreshold(Theta + U, self.gamma/rho)
            U = U + Theta - Z
            if (j % 100 == 0) and (not self.quite):
                print j, numpy.linalg.norm(prevTheta - Theta, 2)
            rho = min(rho*1.1, maxrho)
        W = Theta
        return W

    def predict(self, X):
        return X.dot(self.W)

def test1():
    n = 30
    d = 60
    k = 14
    W = numpy.random.normal(size=(d, k))
    W[:, :4] += numpy.random.normal(0, 10, size=(d, 1))
    W[:, 5:10] += numpy.random.normal(0, 10, size=(d, 1))
    print W[:5,:5]
    X = numpy.random.uniform(-1, 1, size=(n, d))
    y = X.dot(W)
    mssl = pMSSL(max_epochs=200, quite=True)
    MSE = numpy.zeros((5,5))
    for j, g in enumerate(10**numpy.linspace(-1, 3, 5)):
        for i, l in enumerate(10**numpy.linspace(-1, 3, 5)):
            try:
                mssl.train(X[:20], y[:20], rho=1e-4, gamma=g, lambd=l)
            except:
                print "Pass -- Lamdba %f, Gamma %f" % (l, g)
                continue
            yhat = mssl.predict(X[20:])
            mse = numpy.mean((yhat - y[20:])**2)
            MSE[i, j] = mse
            try:
                if mse < bestmse:
                    bestmssl = mssl
            except NameError:
                bestmssl = mssl

    pyplot.subplot(2,1,1)
    pyplot.imshow(MSE, interpolation="none")
    pyplot.subplot(2,1,2)
    pyplot.imshow(bestmssl.Omega, interpolation="none", cmap="Reds")

def test2():
    n = 60
    d = 30
    k = 14
    l = 1e-3
    g = 1e4

    W = numpy.random.normal(size=(d, k))
    W[:, :4] += numpy.random.normal(0, 10, size=(d, 1))
    W[:, 5:10] += numpy.random.normal(0, 10, size=(d, 1))
    print W[:5,:15]
    X = numpy.random.uniform(-1, 1, size=(n, d))
    y = X.dot(W)
    mssl = pMSSL(max_epochs=200, quite=True)
    mssl.train(X, y, rho=1e-4, gamma=g, lambd=l)
    mssl.Omega[numpy.abs(mssl.Omega) < 1e-10] = 0
    print mssl.W[:5, :15]
    #pyplot.imshow(numpy.linalg.inv(mssl.Omega))
    pyplot.imshow(mssl.Omega, interpolation="none", cmap="Reds")
    pyplot.title("Lamdba %f, Gamma %f" % (l, g))
    pyplot.show()

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
    test1()
    #climatetest()

