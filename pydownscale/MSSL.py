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
import utils
from joblib import Parallel, delayed
from sklearn import preprocessing
EPSABS = 1e-2
EPSREL = 1e-4

def hascomplex(x):
    if numpy.iscomplex(x).sum() > 0:
        return True
    else:
        return False

def softthreshold(X, thres):
    return numpy.piecewise(X, 
        [X > thres, numpy.abs(X) <= thres, X < -thres], 
        [lambda X: X - thres, 0, lambda X: X+thres])

class pMSSL:
    def __init__(self, max_epochs=1000, quiet=True, lambd=1e-3, 
        gamma=1e-3, walgo='admm', omega_epochs=100, w_epochs=100, 
        rho=1., mpicomm=None, mpiroot=None, ytransform=None, num_proc=1,
        xtransform=None, how='regression'):
        self.lambd = float(lambd)
        self.max_epochs = max_epochs
        self.quiet = quiet
        self.gamma = gamma
        self.walgo = walgo
        self.omega_epochs = omega_epochs
        self.w_epochs = w_epochs
        self.rho = rho
        self.mpicomm = mpicomm
        self.mpiroot = mpiroot
        self.ytransform = ytransform
        self.xtransform = xtransform
        self.num_proc = num_proc
        self.how = how
        if (self.how == 'classify') and (walgo != 'multiprocessor'):
            # Multiprocessor is the only W update that can classify
            self.walgo = 'multiprocessor'
            self.ytransform = None  # Y inputs should not be scaled

    def fit(self, X, y, epsomega=1e-3, epsw=1e-3):
        start_time = time.time()
        Xstart = X.copy()
        if self.xtransform is not None:
            self.xtransform.fit(X)
            X = self.xtransform.transform(X)

        X, self.X_frob = utils.center_frob(X)
        if self.ytransform is not None:
            self.ytransform.fit(y)
            y = self.ytransform.transform(y)

        Xy = X.T.dot(y)
        # Store the number of tasks, samples, and dimensions
        self.K = y.shape[1]
        self.n = y.shape[0]
        self.d = X.shape[1]

        if (self.walgo == 'admm') and (not hasattr(self, 'W')):
            self.W = WADMM(rho=self.rho, gamma=self.gamma, lambd=self.lambd, 
                shape=(self.d, self.K), quiet=self.quiet, w_epochs=self.w_epochs)

        elif (self.walgo == 'mpi') and (not hasattr(self, 'W')):
            self.W = WMPIDistributed(rho=self.rho, gamma=self.gamma, 
                lambd=self.lambd, shape=(self.d, self.K), 
                maxepoch=self.w_epochs, quiet=self.quiet,
                mpicomm=self.mpicomm, root=self.mpiroot)

        elif (self.walgo == 'fista') and (not hasattr(self, 'W')):
            self.W = WFista(self.gamma, self.lambd, shape=(self.d, self.K), 
                maxepochs=self.w_epochs, epscost=1e-4)

        elif (self.walgo == 'multiprocessor') and (not hasattr(self, 'W')):
            self.W = WADMMMultiProcessor(rho=self.rho, gamma=self.gamma, 
                lambd=self.lambd, shape=(self.d, self.K), 
                maxepoch=self.w_epochs, quiet=self.quiet,
                num_proc=self.num_proc, how=self.how)

        if not hasattr(self, 'Omega'):
            self.Omega = OmegaADMM(rho=self.rho, gamma=self.gamma, lambd=self.lambd, K=self.K)

        print "Number of tasks: %i, Number of dimensions: %i, Number of observations: %i, Lambda: %0.4f, Gamma: %0.4f" % (self.K, self.d, self.n, self.lambd, self.gamma)

        WList = [self.W.values.copy()]
        for t in range(self.max_epochs):
            # Update W
            self.W.update(X, y, Omega=self.Omega.values)
            if self.walgo == 'mpi':
                self.mpicomm.Barrier()
            WList.append(self.W.values.copy())

            # Update Omega
            if self.walgo == 'mpi':
                if self.mpicomm.Get_rank() == self.mpiroot:
                    self.Omega.update(X, y, W=self.W.values)
                self.Omega = self.mpicomm.bcast(self.Omega, root=self.mpiroot)
            else:
                self.Omega.update(X, y, W=self.W.values) 

            # Compute difference + convergence
            omega_diff = numpy.linalg.norm(self.Omega.values - self.Omega.prev_omega, 2)
            w_diff = numpy.linalg.norm(self.W.prev_w - self.W.values, 2)
            if (omega_diff < epsomega) and (w_diff < epsw):
                break

            # Print stuff?
            if not self.quiet:
                print "FULL Iteration %i, w zeros: %i, omega zeros: %i, lambda: %2.4f, gamma: %2.4f"  % (t, numpy.sum(self.W.values == 0), numpy.sum(self.Omega.values == 0),
                       self.lambd, self.gamma)
                print "Omega diff:", omega_diff, "\tW Diff:", w_diff

            # 24 Hours is almost up
            if (time.time() - start_time) > (20. * 60 * 60):
                break
            if self.walgo == 'mpi':
                self.mpicomm.Barrier()

        minutes = (time.time() - start_time) / 60.
        print "Gamma: %0.4f, Lambda: %0.4f, Converged in %i iterations, %2.3f Minutes" % (self.gamma, self.lambd, t, minutes)

    def cost(self, X, y, W, Omega):
        cost = 0.5*numpy.linalg.norm(y - X.dot(W), 2)
        cost += self.lambd*numpy.trace(W.dot(Omega).dot(W.T))
        if numpy.sum(Omega) != 0:
            cost -= self.K/2.*numpy.log(numpy.linalg.det(Omega))
        cost += self.lambd * numpy.linalg.norm(Omega, 1)
        return cost 

    def predict(self, X):
        if self.xtransform is not None:
            X = self.xtransform.transform(X)
        X = X.dot(self.X_frob)
        yhat = self.W.predict(X)

        if self.ytransform is not None:
           yhat = self.ytransform.inverse_transform(yhat)
        return yhat


class OmegaADMM:
    def __init__(self, gamma, lambd, rho, K, omega_epochs=100, quiet=True):
        self.gamma = gamma
        self.lambd = lambd
        self.rho = rho
        self.values = numpy.eye(K)
        self.quiet = quiet
        self.omega_epochs = omega_epochs

    def update(self, X, y, W):
        K = y.shape[1]
        self.prev_omega = self.values.copy()
        Theta = self.values.copy()
        Z = self.values.copy() 
        U = numpy.zeros(shape=(K, K))
        j = 0
        resid = []
        # Sp = self.W.T.dot(self.W)   # why is this in the MSSL paper?
        S = numpy.cov(W.T)

        for j in range(self.omega_epochs):
            try:
                L, Q = numpy.linalg.eigh(self.rho * (Z - U) - S)
            except numpy.linalg.LinAlgError:
                pass

            # Updates
            Theta_tilde = numpy.eye(K)
            numpy.fill_diagonal(Theta_tilde, (L + numpy.sqrt(L**2 + 4*self.rho))/(2*self.rho))
            Theta = Q.dot(Theta_tilde).dot(Q.T)
            Z_prev = Z.copy()
            Z = softthreshold(Theta + U, self.lambd/self.rho)
            U = U + Theta - Z

            # check for convergence
            dualresid = numpy.linalg.norm(-self.rho * (Z - Z_prev), 'fro')
            primalresid = numpy.linalg.norm(Theta - Z, 'fro')
            epspri = X.shape[0] * EPSABS + EPSREL * numpy.max([numpy.linalg.norm(Theta, 'fro'), numpy.linalg.norm(Z, 'fro'), 0])
            epsdual = X.shape[0] * EPSABS + EPSREL * numpy.linalg.norm(self.rho*U, 'fro')

            if not self.quiet:
                print "omega update:", j, "Dualresid:", dualresid, "Dual EPS:", epsdual

            if (dualresid < epsdual) and (primalresid < epspri):
                break

            if j == (self.omega_epochs-1):
                print "OMEGA DID NOT CONVERGE"

        self.values = Z 


class WFista:
    def __init__(self, gamma, lambd, shape, maxepochs=100, epscost=1e-4):
        self.gamma = gamma
        self.lambd = lambd
        self.values = numpy.zeros(shape)
        self.maxepochs = maxepochs
        self.epscost = epscost

    def update(self, X, y, Omega):
        Theta = self.values.copy()
        tk = 1/(numpy.linalg.norm(X.T.dot(X), 1)) # tk exits in (0, 1/||X.T*X||)
        # cache these
        XX = X.T.dot(X)
        XY = X.T.dot(y)
        for j in range(self.maxepochs):
            cost, gmat = self._w_cost(X, y, Theta, Omega, XX=XX, XY=XY)
            W = self.shrinkage_threshold(W - tk*gmat, alpha=self.gamma*tk)
            if j == 0:
                costdiff = numpy.abs(cost)
            else:
                costdiff = numpy.abs(costprev - cost)
            costprev = cost
            if costdiff < self.epscost:
                break

        self.values = Theta

        # Lets parallelize this
    def _w_cost(self, X, y, Theta, Omega, XX=None, XY=None):
        XW = X.dot(Theta)
        if XX is None:
            XX = X.T.dot(X)
        if XY is None:
            XY = X.T.dot(y)
        f = 0.5*numpy.linalg.norm(y - XW, 2)
        f += self.lambd*numpy.trace(W.dot(Omega).dot(W.T))
        gmat = (XX.dot(W) - XY)/len(y)  # the gradients
        gmat += 2*W.dot(Omega) # *self.lambd
        return numpy.sum(f), gmat


    def shrinkage_threshold(self, a, alpha):
        return numpy.maximum(numpy.zeros(shape=a.shape), a-alpha) - numpy.maximum(numpy.zeros(shape=a.shape), -a-alpha)

    def predict(self, X):
        return X.dot(self.values)


class WMPIDistributed:
    def __init__(self, lambd, gamma, shape, rho=1., 
                maxepoch=100, maxepoch_inner=100, quiet=True, 
                mpicomm=None, root=0):
        self.lambd = lambd
        self.gamma = gamma
        self.rho = rho
        self.maxepoch = maxepoch
        self.maxepoch_inner = maxepoch_inner
        self.quiet = quiet
        self.values = numpy.zeros(shape=shape)
        self.mpicomm = mpicomm
        self.root = root 
        self.size = mpicomm.Get_size() 
        self.curr_rank = self.mpicomm.Get_rank()

    def update(self, X, Y, Omega):
        ## MPI SETTINGS
        self.prev_w = self.values.copy()
        nsamples, nfeatures = X.shape
        _, ntasks = Y.shape
        Theta = self.values.copy()
        if self.curr_rank == self.root:
            feature_split = numpy.array_split(numpy.arange(nfeatures), self.size)
        else:
            feature_split = None

        self.feat = self.mpicomm.scatter(feature_split)
        feature_split = numpy.array_split(numpy.arange(nfeatures), self.size)
        XWbar = self.XW_mean(X, Theta, feature_split)

        if hasattr(self, 'Zbar_prev'):
            Zbar = self.Zbar_prev
        else:
            Zbar = numpy.zeros(shape=Y.shape) #XWbar.copy() 
        U = numpy.zeros(shape=Y.shape)

        for k in range(self.maxepoch):
            Zbar_prev = Zbar.copy()

            # compute that update
            xi = X[:, self.feat]
            thetai = Theta[self.feat, :].copy()
            b = xi.dot(thetai) + Zbar - XWbar - U
            thetai = self._w_update(xi, thetai, Omega=Omega, b=b)
            # Collect theta updates to root
            theta_updates = self.mpicomm.gather([self.feat, thetai], root=self.root) 
            if self.curr_rank == self.root:
                for feat, ti in theta_updates:
                    Theta[feat, :] = ti

                # Compute Z and U updates
                XWbar = self.XW_mean(X, Theta, feature_split)
                temp = self.size * self.rho * (XWbar + U) + self.size*Y
                Zbar = temp / (self.size**2 + self.rho * self.size)
                U = U + XWbar - Zbar

            # Broadcast updates to all ranks
            Theta = self.mpicomm.bcast(Theta, root=self.root)
            XWbar = self.mpicomm.bcast(XWbar, root=self.root)
            Zbar = self.mpicomm.bcast(Zbar, root=self.root)
            U = self.mpicomm.bcast(U, root=self.root)

            # compute residuals and check for convergence
            dualresid = numpy.linalg.norm(-self.rho * (Zbar - Zbar_prev), 'fro')
            primalresid = numpy.linalg.norm(XWbar - Zbar, 'fro')
            epspri = numpy.sqrt(nsamples*ntasks) * EPSABS   + EPSREL * numpy.max([numpy.linalg.norm(XWbar, 'fro'), numpy.linalg.norm(Zbar, 'fro'), 0])
            epsdual = numpy.sqrt(nsamples*ntasks) * EPSABS + EPSREL * numpy.linalg.norm(self.rho*U, 'fro')

            if (not self.quiet) and (self.curr_rank == self.root):
                print "RANK: %i, Iteration %i, PrimalResid: %2.2f, EPSPRI: %2.2f, DualResid: %2.2f"\
                "EPSDUAL: %2.2f" % (self.root, k, primalresid, epspri,  dualresid, epsdual)
            if (epspri > primalresid) and (epsdual > dualresid):
                break

            if (k > 0) and (prevdual < dualresid):
                break
            prevdual = dualresid

        self.Zbar_prev = Zbar_prev
        self.values = Theta 

    def _w_update(self, x, w, Omega, b):
        theta = w.copy()
        n = x.shape[0]
        z = w.copy() #numpy.zeros(shape=w.shape)
        u = numpy.zeros(shape=w.shape)

        # cache multiplications
        xb = x.T.dot(b)
        xx = x.T.dot(x)
        for l in range(self.maxepoch_inner):
            zprev = z.copy()

            # updates
            theta = solve_sylvester(self.rho*xx + self.rho * numpy.eye(xx.shape[0]), 2*Omega, self.rho*xb + self.rho*(z-u))
            z = softthreshold(theta + u, 1.*self.gamma/self.rho)
            u = u + theta - z
            # compute residuals
            dualresid = numpy.linalg.norm(-self.rho * (z - zprev), 'fro')
            primalresid = numpy.linalg.norm(theta - z, 'fro')
            epspri = n * EPSABS  + EPSREL * numpy.max([numpy.linalg.norm(theta, 'fro'), numpy.linalg.norm(z, 'fro'), 0])
            epsdual = n * EPSABS + EPSREL * numpy.linalg.norm(self.rho*u, 'fro')

            # check for convergence 
            if (dualresid < epsdual) and (primalresid < epspri):
                break

        return z

    def XW_mean(self, X, W, feature_split):
        temp = [X[:,features].dot(W[features,:]) for features in feature_split]
        return reduce(numpy.add, temp) / self.size

    def predict(self, X):
        return X.dot(self.values)

class WADMM:
    def __init__(self, gamma, lambd, rho, shape, quiet=True, w_epochs=100):
        self.gamma = gamma
        self.lambd = lambd
        self.rho = rho
        self.values = numpy.zeros(shape=shape)
        self.quiet = quiet
        self.w_epochs = w_epochs

    def update(self, X, y, Omega):
        # save the previous W
        self.prev_w = self.values.copy()
        # Initialize Updates
        Theta = self.values.copy()      # dxk
        Z = self.values.copy()  ## warm start
        U = numpy.zeros(shape=Z.shape)

        # Cache multiplication
        XX = X.T.dot(X)  # dxd
        Xy = X.T.dot(y)  # dxk

        for j in range(self.w_epochs): 
            Z_prev = Z.copy()

            # Update Theta
            C = Xy + self.rho * (Z - U)
            Theta = solve_sylvester(XX + self.rho * numpy.eye(XX.shape[0]), 2*Omega, C)
            Z = softthreshold(Theta + U, self.gamma/self.rho)
            U = U + Theta - Z

            # Compute residuals 
            dualresid = numpy.linalg.norm(-self.rho*(Z - Z_prev), 2)
            primalresid = numpy.linalg.norm(Theta - Z, 2)
            epspri = X.shape[0] * EPSABS + EPSREL * numpy.max([numpy.linalg.norm(Theta, 2), numpy.linalg.norm(Z, 2), 0])
            epsdual = X.shape[0] * EPSABS + EPSREL * numpy.linalg.norm(self.rho*U,2)

            # Confirm the dual is decreasing
            if (j > 0) and (prevdual < dualresid):
                #break
                pass
            prevdual = dualresid
            # Check for convergence
            if (dualresid < epsdual) and (primalresid < epspri):
                break

            if not self.quiet:
                print "Learning W, Iteration: %i, Rho: %2.2f, Dualresid: %2.4f, EPS_Dual: %2.4f, PriResid: %2.4f, EPS_Pri: %2.4f" % (j, self.rho, dualresid, epsdual, primalresid, epspri)

        self.values = Z

    def predict(self, X):
        return X.dot(self.values)

        

#thetai = Theta[self.feat, :].copy()
def compute_b(xi, thetai, Zbar, XWbar, U):
    return xi.dot(thetai) + Zbar - XWbar - U

def _w_update(x, w, Omega, b, rho, gamma, maxepochs=25):
    theta = w.copy()
    n = x.shape[0]
    z = w.copy() #numpy.zeros(shape=w.shape)
    u = numpy.zeros(shape=w.shape)

    # cache multiplications
    xb = x.T.dot(b)
    xx = x.T.dot(x)
    for l in range(maxepochs):
        zprev = z.copy()
        # updates
        theta = solve_sylvester(rho*xx + rho * numpy.eye(xx.shape[0]), 2*Omega, rho*xb + rho*(z-u))
        z = softthreshold(theta + u, 1.*gamma/rho)
        u = u + theta - z
        # compute residuals
        dualresid = numpy.linalg.norm(-rho * (z - zprev), 'fro')
        primalresid = numpy.linalg.norm(theta - z, 'fro')
        epspri = n * EPSABS  + EPSREL * numpy.max([numpy.linalg.norm(theta, 'fro'), numpy.linalg.norm(z, 'fro'), 0])
        epsdual = n * EPSABS + EPSREL * numpy.linalg.norm(rho*u, 'fro')
        # check for convergence 
        if (dualresid < epsdual) and (primalresid < epspri):
            break

    return z

def zbar_loss(zbar, xwbar, y, omega, rho, u, N):
    n, K = xwbar.shape
    zbar = zbar.reshape(n, K)
    loss = 0.5 * numpy.trace(y * (N*zbar - numpy.log(1+numpy.exp(N*zbar))) + (y-1) * numpy.log(1 + numpy.exp(N*zbar)))
    loss += rho / 2 * numpy.linalg.norm(zbar - xwbar + u, 'fro')
    return -loss

def zbar_gradient(zbar, xwbar, y, omega, rho, u, N):
    n, K = xwbar.shape
    zbar = zbar.reshape(n, K)
    sig_s = sigmoid(N*zbar)
    #grad = 0.5 * X.dot(X.T).dot(sig_s - y)
    grad = 0.5 * (sig_s - y)
    grad += rho * (zbar - xwbar + u)
    return grad.flatten()

def sigmoid(x):
    x[x>20] = 20
    x[x<-20] = -20
    return 1. / ( 1 + numpy.exp(-x))


class WADMMMultiProcessor:
    def __init__(self, lambd, gamma, shape, rho=1., 
                maxepoch=100, maxepoch_inner=25, quiet=True, 
                mpicomm=None, root=0, num_proc=1, how='regression'):
        self.lambd = lambd
        self.gamma = gamma
        self.rho = 1.*rho
        self.maxepoch = maxepoch
        self.maxepoch_inner = maxepoch_inner
        self.quiet = quiet
        self.values = numpy.zeros(shape=shape)
        self.num_proc = num_proc
        self.size = num_proc
        self.how = how
    
    def update(self, X, Y, Omega):
        ## Intialize Variables
        self.prev_w = self.values.copy()
        nsamples, nfeatures = X.shape
        _, ntasks = Y.shape
        Theta = self.values.copy()
        feature_split = numpy.array_split(numpy.arange(nfeatures), self.num_proc)
        XWbar = self.XW_mean(X, Theta, feature_split)

        if hasattr(self, 'Zbar_prev'):
            Zbar = self.Zbar_prev
        else:
            Zbar = numpy.zeros(shape=Y.shape) #XWbar.copy() 
        U = numpy.zeros(shape=Y.shape)

        for k in range(self.maxepoch):
            Zbar_prev = Zbar.copy()
            
            # compute the theta update
            bs = Parallel(n_jobs=self.num_proc)(
                delayed(compute_b)(X[:,feat], Theta[feat,:], Zbar, XWbar, U) for feat in feature_split)
            temptheta = Parallel(n_jobs=self.num_proc)(
                delayed(_w_update)(X[:, feat],Theta[feat,:], Omega=Omega, b=bs[i],
                                   rho=self.rho, gamma=self.gamma, maxepochs=self.maxepoch_inner)
                for i, feat in enumerate(feature_split))
            Theta = numpy.vstack(temptheta)

            # Compute Z and U updates
            XWbar = self.XW_mean(X, Theta, feature_split)
            if self.how == 'regression':
                temp =  self.rho * (XWbar + U) + Y
                Zbar = temp / (self.size + self.rho )
            elif self.how =='classify':
                Zbar = minimize(zbar_loss, Zbar_prev, jac=zbar_gradient, args=(XWbar, Y, Omega, self.rho, U, self.num_proc)).x
                #Zbar = minimize(zbar_loss, Zbar_prev, args=(XWbar, Y, Omega, self.rho, U)).x
                Zbar = Zbar.reshape(nsamples, ntasks) / self.num_proc
                
            # compute residuals and check for convergence
            dualresid = numpy.linalg.norm(-self.rho * (Zbar - Zbar_prev), 'fro')
            primalresid = numpy.linalg.norm(XWbar - Zbar, 'fro')
            epspri = numpy.sqrt(nsamples*ntasks) * EPSABS   + EPSREL * numpy.nanmax([numpy.linalg.norm(XWbar, 'fro'), numpy.linalg.norm(Zbar, 'fro'), 0])
            epsdual = numpy.sqrt(nsamples*ntasks) * EPSABS + EPSREL * numpy.linalg.norm(self.rho*U, 'fro')

            if self.how == 'classify':
                self.values = Theta.copy()
                y_hat = self.predict(X)
                print roc_auc_score((Y > 0.5).flatten(), y_hat.flatten())
        
            if (not self.quiet): #  and (self.curr_rank == self.root):
                print "Iteration %i, PrimalResid: %2.2f, EPSPRI: %2.2f, DualResid: %2.2f"\
                "EPSDUAL: %2.2f" % (k, primalresid, epspri,  dualresid, epsdual)
            if (epspri > primalresid) and (epsdual > dualresid):
                break

            if (k > 0) and (prevdual < dualresid):
                break
            prevdual = dualresid


        self.Zbar_prev = Zbar_prev
        self.values = Theta.copy() 

    def XW_mean(self, X, W, feature_split):
        temp = [X[:,features].dot(W[features,:]) for features in feature_split]
        return reduce(numpy.add, temp) / self.size

    def predict(self, X):
        if self.how == 'regression':
            return X.dot(self.values)
        elif self.how == 'classify':
            return sigmoid(X.dot(self.values))



if __name__ == "__main__":
    from mssl_tests import test1
    print test1()

