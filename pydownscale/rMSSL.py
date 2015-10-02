__author__ = 'tj'
import numpy
from matplotlib import pyplot
from data import DownscaleData, read_nc_files
from downscale import DownscaleModel
import sys

class pMSSL:
    def __init__(self, max_epochs=1000):
        self.max_epochs = max_epochs

    def train(self,X,y,lambd=1., rho=1.):
        self.X = X
        self.y = y
        self.rho = rho
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
        while (costdiff > 10e-3) and (t < self.max_epochs):
            self.W = self._w_update()
            self.Omega = self._omega_update(self.Omega)
            curr_cost, _ = self._w_cost(self.W)
            costs.append(curr_cost)
            if t == 0:
                costdiff = curr_cost
                prevcost = curr_cost
            else:
                costdiff = numpy.abs(prevcost - curr_cost)
                prevcost = curr_cost
            t += 1
            print "iteration %i, costdiff: %f" % (t, costdiff)


    def shrinkage_threshold(self, a, alpha):
        A = (numpy.abs(a) - alpha)
        idx = numpy.where(A < numpy.zeros(shape=a.shape))
        A[idx] = 0
        return A * numpy.sign(a)

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

    def _omega_update(self, Omega):
        Z = numpy.zeros(shape=(self.K, self.K))
        U = numpy.zeros(shape=(self.K, self.K))
        j = 0
        dualresid = 10e6
        resid = []
        while (dualresid > 0.01) or (j < 10): # force 10 iterations,
            S = self.W.T.dot(self.W)
            L, Q = numpy.linalg.eig(self.rho * (Z - U) - S)
            Omega_tilde = numpy.eye(self.K)
            numpy.fill_diagonal(Omega_tilde, (L + numpy.sqrt(L**2 + 4*self.rho))/(2*self.rho))
            Omega = Q.dot(Omega_tilde).dot(Q.T)
            Z_prev = Z.copy()
            Z = self.softthreshold(Omega + U, self.lambd/self.rho)
            U = U + Omega - Z
            dualresid = numpy.linalg.norm(self.rho * self.X.T.dot(self.y).dot(Z - Z_prev), 2)
           # dualresid = numpy.linalg.norm(self.rho * (Z - Z_prev), 2)
            Z = self.softthreshold(Omega + U, self.lambd/self.rho)
            if j % 500 == 1:
                print "omega update:", j, "Dualresid:", dualresid
            j+=1
            resid.append(dualresid)
            if j % 100 == 0:
                #pyplot.plot(resid)
                #pyplot.draw()
                #pyplot.show()
                resid = []

        return Omega

    def _w_update(self, tk=1.):
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
            #if j % 100 == 1:
            #    print "W update:", j, ", Cost diff:", costdiff
            if j > 1000:
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

    def predict(self, X):
        return X.dot(self.W)

def test1():
    n = 60
    d = 30
    k = 14
    W = numpy.random.normal(size=(d,k))
    W[:, :4] += numpy.random.normal(0, 5, size=(d,1))
    W[:, 5:10] += numpy.random.normal(0, 5, size=(d,1))
    X = numpy.random.uniform(size=(n,d))
    y = X.dot(W)
    mssl = pMSSL()
    mssl.train(X,y)
    #pyplot.imshow(numpy.linalg.inv(mssl.Omega))
    pyplot.imshow(mssl.Omega)
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


