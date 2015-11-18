import numpy
from matplotlib import pyplot
import sys
#numpy.random.seed(0)
from sklearn.cross_validation import KFold
from joblib import Parallel, delayed
'''
Notes:
LOOP Through  EPS values where eps=2**i for i=-8, -9, ..., -32
'''

def qrnn_job(X_train, y_train, X_test, y_test, ntrails=5,  maxepochs=1e4, lr=0.1, tau=0.5,
                 hidden_nodes=4,  tol=1e-5, lower=-numpy.inf):
    qrnn = QRNN(ntrails=ntrails, lr=lr, maxepochs=maxepochs,  tau=tau,
                 hidden_nodes=hidden_nodes,  tol=tol, lower=lower)
    qrnn.fit(X_train, y_train)
    yhat = qrnn.predict(X_test)
    mse = numpy.mean((yhat - y_test)**2)
    return mse, qrnn

class QRNNCV:
    def __init__(self, cv=4, ntrails=5, learning_rate=0.5, maxepochs=1e4,  penalty=0.0, lr=0.1, tau=0.5,
                 hidden_nodes=list([4]),  tol=1e-5, lower=-numpy.inf, n_jobs=2):
        self.lr = learning_rate
        self.maxepochs = maxepochs
        self.layers = 2
        self.penalty = penalty
        self.tau = tau
        self.hidden_nodes = hidden_nodes
        self.lr = lr
        self.lower = lower
        self.tol = tol
        self.ntrails = ntrails
        self.cv = cv
        self.n_jobs = n_jobs

    def fit(self, X, y):
        if isinstance(self.hidden_nodes, int):
            self.hidden_nodes = [self.hidden_nodes]

        cvdata = numpy.array([self._fit(X, y, h) for h in self.hidden_nodes])
        minidx = numpy.argmin(cvdata[:, 1])
        print "Chosen Number of Nodes: %i" % cvdata[minidx,0]
        mse, self.qrnn = self._single_fit(X, y, X, y, cvdata[minidx,0])
        print cvdata


    def _fit(self, X, y, nodes):
        nobs = len(y)
        kfold = KFold(nobs, n_folds=self.cv)
        jobs = (delayed(qrnn_job)(X[trainidx], y[trainidx], X[testidx], y[testidx], hidden_nodes=nodes,
                lr=self.lr, maxepochs=self.maxepochs,  tau=0.5,
                tol=1e-5, lower=-numpy.inf) for trainidx, testidx in kfold)
        mse = Parallel(n_jobs=self.n_jobs)(jobs)

        return nodes, numpy.mean([val[0] for val in mse])

    def _single_fit(self, Xtrain, ytrain, Xtest, ytest, hidden_nodes):
        qrnn = QRNN(lr=self.lr, maxepochs=self.maxepochs,  penalty=self.penalty, tau=0.5,
                 hidden_nodes=hidden_nodes,  tol=1e-5, lower=-numpy.inf)
        qrnn.fit(Xtrain, ytrain)
        yhat = qrnn.predict(Xtest)
        mse = numpy.mean((yhat - ytest)**2)
        return mse, qrnn


    def predict(self, X):
        return self.qrnn.predict(X)



class QRNN:
    def __init__(self, maxepochs=1e4,  penalty=0.0, lr=0.01, tau=0.5,
                 hidden_nodes=4,  tol=1e-5, lower=-numpy.inf, ntrails=5,
                 momentum=0.5):
        self.lr = lr
        self.maxepochs = int(maxepochs)
        self.layers = 2
        self.penalty = penalty
        self.tau = tau
        self.hidden_nodes = hidden_nodes
        self.lr = lr
        self.lower = lower
        self.tol = tol
        self.ntrails = ntrails
        self.momentum = momentum

    def fit(self, X, y):
        self.EPS = 2.**(-numpy.arange(8, 33))
        self.X = X

        if len(y.shape) == 1:
            self.y = y[:, numpy.newaxis]
        else:
            self.y = y

        for j in range(self.ntrails):
            self._fit()
            print j
            if (j == 0) or (self.lowcost < bestcost):
                bestw = self.w
                bestcost = self.lowcost

            self.w = bestw


    def _fit(self):
        self.init_weights()
        deltaw0 = numpy.zeros(self.w[0].shape)
        deltaw1 = numpy.zeros(self.w[1].shape)
        cost = self.cost(self.y, self.EPS[0])
        for eps in self.EPS:
            for j in range(self.maxepochs):
                prevcost = cost
                err = self.y - self.feedforward(self.X, eps)
                grad1, grad2 = self.gradient(self.X, err, eps)

                self.w[0] -= self.lr*grad1 + self.momentum * deltaw0
                self.w[1] -= self.lr*grad2 + self.momentum * deltaw1

                deltaw0 = self.lr*grad1.copy()
                deltaw1 = self.lr*grad2.copy()

                cost = self.cost(self.y, eps)
                if numpy.abs(cost - prevcost) < self.tol:
                    break

        self.lowcost = cost


    def init_weights(self):
        self.w = []
        numout = self.y.shape[1]
        features = self.X.shape[1]
        self.w.append(numpy.random.uniform(-0.5, 0.5, size=(features+1, self.hidden_nodes)))
        self.w.append(numpy.random.uniform(-0.5, 0.5, size=(self.hidden_nodes+1, numout)))

    def cost(self, y, eps):
        yhat = self.feedforward(self.X, eps)
        err = y - yhat
        cost = numpy.mean(self.tilted_huber(err, eps))
        #cost += self.penalty * numpy.sum(self.w[0][:-1]**2) / (self.w[0].size - self.w[0].shape[1])
        return cost

    def feedforward(self, X, eps):
        X = numpy.column_stack((X, numpy.ones(X.shape[0])))
        self.h1 = X.dot(self.w[0])
        y1 = self.tanh(self.h1)
        self.y1aug = numpy.column_stack((y1, numpy.ones(y1.shape[0])))
        self.h2 = self.y1aug.dot(self.w[1])
        return self.hramp(self.h2, eps)

    def gradient(self, X, err, eps):
        X = numpy.column_stack((X, numpy.ones(X.shape[0])))

        delta2 = self.hramp_prime(self.h2, eps)[:, numpy.newaxis] * self.tilted_huber_prime(err, eps)
        gradient2 = -self.y1aug.T.dot(delta2)/err.size  ## penalty2 is zero

        err1 = delta2.dot(self.w[1][:-1].T)
        delta1 = err1*self.tanh_prime(self.h1)

        gradient1 = -(X.T.dot(delta1))/err.size
        return gradient1, gradient2

    def predict(self, X):
        return self.feedforward(X,  self.EPS[-1]).flatten()

    def tanh(self, x):
        return numpy.tanh(0.5*x)

    def tanh_prime(self, x):
        return 0.5 * (1 - numpy.tanh(0.5*x)**2)

    def huber(self, u, eps):
        u = numpy.abs(u)
        return numpy.piecewise(u, [u <= eps, u > eps, numpy.isnan(u)],
                        [lambda k: k**2/(2*eps),
                         lambda k: numpy.abs(k) - eps/2,
                         0])

    def huber_prime(self, u, eps):
        dh = u.copy()/eps
        dh[u > eps] = 1
        dh[u < -eps] = -1
        dh[numpy.isnan(u)] = 0
        return dh

    def hramp(self, u, eps):
        if self.lower == -numpy.inf:
            return u
        else:
            h = numpy.zeros(u.shape)
            gt = u >= self.lower
            lt = u <= self.lower
            h[gt] = self.huber(u[gt]-self.lower, eps=eps)
            h[lt] = 0
            h += self.lower
            return h

    def hramp_prime(self, x, eps):
        if self.lower == -numpy.inf:
            return numpy.ones(x.shape[0])
        else:
            dhr = (x - self.lower)/eps
            dhr[x > (self.lower + eps)] = 1
            dhr[x < self.lower] = 0
            return dhr

    def tilted_huber(self, x, eps):
        return numpy.piecewise(x, [x > 0, x <= 0],
                               [lambda x: self.tau * self.huber(x, eps=eps),
                                lambda x: (1-self.tau) * self.huber(x, eps=eps)])

    def tilted_huber_prime(self, x, eps):
        return numpy.piecewise(x, [x > 0, x <= 0],
            [lambda k: self.tau * self.huber_prime(k, eps),
             lambda k: (1-self.tau) * self.huber_prime(k, eps)])





if __name__ == "__main__":
    from scipy.stats import pearsonr, spearmanr
    from sklearn import datasets
    data = datasets.load_boston()
    X = data["data"]
    y = data["target"]
    ntrain = 450

    xcenter = X[:ntrain].mean(axis=0)
    xscale = X[:ntrain].std(axis=0)
    ycenter = y[:ntrain].mean(axis=0)
    yscale = y[:ntrain].std(axis=0)

    X = (X - X[:ntrain].mean(axis=0)) / X[:ntrain].std(axis=0)
    y = (y - y[:ntrain].mean(axis=0)) / y[:ntrain].std(axis=0)

    Xtrain = X[:ntrain]
    ytrain = y[:ntrain]
    Xtest = X[ntrain:]
    ytest = y[ntrain:]
    import time
    t0 = time.time()
    model = QRNNCV(tol=1e-7, hidden_nodes=range(1, 11), ntrails=5, n_jobs=8, cv=8)
    model.fit(Xtrain, ytrain)
    print "time", time.time() - t0

    ytestunscaled = ytest * yscale + ycenter
    yhat = model.predict(Xtest) * yscale + ycenter
    yhat = yhat.flatten()

    print "y:", ytestunscaled[:10]
    print "Yhat", yhat[:10]

    print "Pearson:", pearsonr(ytestunscaled, yhat)[0]
    print "Spearman:", spearmanr(ytestunscaled, yhat)[0]

    print "MSE:", numpy.mean((ytestunscaled-yhat)**2)

    pyplot.plot(yhat, color="red")
    pyplot.plot(ytestunscaled, color="blue")
    pyplot.show()