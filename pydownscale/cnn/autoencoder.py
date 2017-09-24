
# coding: utf-8

# In[123]:

import pickle
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

import sys
import os

sys.path.append("/Users/tj/repos/pydownscale/")
from pydownscale.MSSL import pMSSL
from pydownscale.data import DownscaleData
from pydownscale.utils import RootTransform
from pydownscale import config

config.save_dir = '/Users/tj/Downloads/'

class ClimateData:
    def __init__(self):
        #data_dir = "/gss_gpfs_scratch/vandal.t/sadm-experiments/DownscaleData"
        data_dir = "/Users/tj/Downloads/"
        #data = pickle.load(open(os.path.join(data_dir, "newengland_D_12781_8835.pkl"), "r"))
        data = pickle.load(open(os.path.join(data_dir, "newengland_D_731_8835.pkl"), "r"))

        max_train_year = 2004
        seas = "JJA"

        seasonidxs = np.where(data.observations['time.season'] == seas)[0]
        self.X = data.get_X()[seasonidxs]
        self.Y, self.locs = data.get_y()
        self.Y = self.Y[seasonidxs]

        self.t = data.observations['time'][seasonidxs]

        train_rows = np.where(self.t['time.year'] <= max_train_year)[0]
        test_rows = np.where(self.t['time.year'] > max_train_year)[0]
        ttest = self.t[test_rows]

        self.Xtrain, self.Xtest = X[train_rows], X[test_rows]
        self.YtrainTrue, self.YtestTrue = Y[train_rows], Y[test_rows]

        self.Xmu, self.Xsd = Xtrain.mean(axis=0), Xtrain.std(axis=0)
        Xtrain = (self.Xtrain - self.Xmu) / self.Xsd
        Xtest = (self.Xtest - self.Xmu) / self.Xsd

        self.scale_y = RootTransform(0.25)
        self.scale_y.fit(YtrainTrue)
        self.Ytrain = scale_y.transform(YtrainTrue)
        self.Ytest = scale_y.transform(YtestTrue)

    def get_data(self, train=True):
        if train:
            return self.Xtrain, self.Ytrain
        else:
            return self.Xtest, self.Ytest



def test_data(n=1000, d=50):
    X = np.random.normal(size=(n*2, d))
    X[:,2:] /= 10.
    y = 2 * X[:,0] - X[:,1]
    Xtrain, ytrain = X[:n], y[:n]
    Xtest, ytest = X[n:], y[n:]
    return Xtrain, ytrain, Xtest, ytest

def generate_batches(X, y, batch_size=20):
    iters = X.shape[0] / batch_size
    idxs = range(X.shape[0])
    np.random.shuffle(idxs)
    for j in range(iters):
        rows = idxs[j*batch_size:(j+1)*batch_size]
        yield X[rows], y[rows]

# In[125]:

class AutoEncoder(object):
    def __init__(self, x, hidden_layers=(100,)):
        self.hidden_layers = hidden_layers

        # set some placeholders
        self.x_ = x
        self.keep_prob = tf.placeholder_with_default(1.0, shape=[])

        self._build()

    def _encoder(self, x):
        for j, h in enumerate(self.hidden_layers):
            if j == (len(self.hidden_layers)-1):
                activation=None
            else:
                activation = tf.nn.relu
            x = tf.layers.dense(x, h, activation=tf.nn.relu, name='enc_%i' % j)
        return x

    def _decoder(self, x):
        for j, h in enumerate(self.hidden_layers[::-1]):
            activation = tf.nn.relu
            x = tf.layers.dense(x, h, activation=activation, name='dec_%i' % j)
        x = tf.layers.dense(x, self.x_.shape[1], name='dec_out')
        return x

    def _build(self):
        self.enc_x = self._encoder(self.x_)
        dec_x = self._decoder(self.enc_x)

        self.loss = tf.losses.mean_squared_error(self.x_, dec_x)
        optimizer = tf.train.AdamOptimizer()
        self.opt = optimizer.minimize(self.loss)

class BaseNetwork(object):
    def __init__(self, x, init_kernels, init_biases):
        self.x = x
        self.init_kernels = init_kernels
        self.init_biases = init_biases
        self.encoded = self._build_graph()

    def _build_graph(self):
        h = self.x
        for j, (w_key, b_key) in enumerate(zip(sorted(self.init_kernels), sorted(self.init_biases))):
            w = tf.constant_initializer(self.init_kernels[w_key])
            b = tf.constant_initializer(self.init_biases[b_key])
            h = tf.layers.dense(h, self.init_kernels[w_key].shape[1], kernel_initializer=w,
                                bias_initializer=b, activation=tf.nn.relu)
        return h



class AEModel(BaseNetwork):
    def __init__(self, x, y, init_kernels, init_biases, n_outputs=1, 
                which='regression'):
        BaseNetwork.__init__(self, x, init_kernels, init_biases)
        self.n_outputs = n_outputs
        self.y = y
        self.which = which
        self._extend_graph()

    def _extend_graph(self):
        if self.which == 'regression':
            self.prediction = tf.layers.dense(self.encoded, self.n_outputs)
            self.loss = tf.losses.mean_squared_error(self.y, self.prediction)
        elif self.which == 'classify':
            self.prediction = tf.layers.dense(self.encoded, self.n_outputs, activation=tf.sigmoid)
            self.loss = tf.losses.sigmoid_cross_entropy(self.y, self.prediction)

        self.opt = tf.train.AdamOptimizer().minimize(self.loss)

def train_autoencoder(Xtrain, Ytrain, Xtest, Ytest):
    with tf.Graph().as_default(), tf.device("/cpu:0"):
        # setup graph
        x = tf.placeholder(tf.float32, shape=[None, Xtrain.shape[1]])
        autoencode = AutoEncoder(x, hidden_layers=(50,10))
        enc_variables = [var for var in tf.trainable_variables() if 'enc' in var.op.name]
        saver = tf.train.Saver()

        # start session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(21):
                losses = []
                for xbatch, ybatch in generate_batches(Xtrain, Ytrain):
                    res = sess.run([autoencode.opt, autoencode.loss], feed_dict={x: xbatch})
                    losses.append(res[1])
                if epoch % 10 == 0:
                    print "Epoch: %i, Loss: %f" % (epoch, np.mean(losses))
                    saver.save(sess, os.path.join(config.save_dir, "autoencoder"))

            enc_kernels, enc_biases = dict(), dict()
            for var in enc_variables:
                if 'kernel' in var.op.name:
                    enc_kernels[var.op.name] = sess.run(var)
                else:
                    enc_biases[var.op.name] = sess.run(var)
    return enc_kernels, enc_biases

def train_supervised(Xtrain, Ytrain, Xtest, Ytest, weights=None, biases=None, which='regression'):
    if (weights is None) or (biases is None):
        weights, biases = train_autoencoder(Xtrain, Ytrain, Xtest, Ytest)

    with tf.Graph().as_default(), tf.device('cpu:0'):
        x = tf.placeholder(tf.float32, shape=[None, Xtrain.shape[1]])
        y = tf.placeholder(tf.float32, shape=[None, 1])
        reg_model = AEModel(x, y, weights, biases, which=which)
        # start session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for epoch in range(101):
                losses = []
                for xbatch, ybatch in generate_batches(Xtrain, Ytrain):
                    res = sess.run([reg_model.opt, reg_model.loss], feed_dict={x: xbatch, y:
                                                                               ybatch[:,np.newaxis]})
                    losses.append(res[1])
                if epoch % 10 == 0:
                    print "Epoch: %i, Loss: %f" % (epoch, np.mean(losses))
                    #saver.save(sess, os.path.join(config.save_dir, "")) 
            test_predictions = sess.run(reg_model.prediction, feed_dict={x: Xtest})
            plt.scatter(Ytest, test_predictions[:,0])
            plt.savefig("test_regression.pdf")


Xtrain, Ytrain, Xtest, Ytest = test_data()
precip_days = Ytrain > 1.
test_precip_days = Ytest > 1.

enc_weights, enc_biases = train_autoencoder(Xtrain, Ytrain, Xtest, Ytest)
train_supervised(Xtrain, precip_days, Xtest, test_precip_days, which='classify',
                weights=enc_weights, biases=enc_biases)
train_supervised(Xtrain[precip_days], Ytrain[precip_days], Xtest[test_precip_days],
                 test_precip_days[test_precip_days], which='regression',
                weights=enc_weights, biases=enc_biases)

# Train Classification
sys.exit()

# In[154]:

cnn_classify = CNNDownscale(sess=tf.InteractiveSession(), how='classify')
ytrain_classify = (Y[train_rows] > 10.)*1.
ytest_classify = (Y[test_rows] > 10.)*1.
print "ytrain_classify", ytrain_classify.shape

cnn_classify.fit(Xtrain, ytrain_classify)
ypred = cnn_classify.predict(Xtest)
print "Test ROC: %g" % roc_auc_score(ytest_classify.flatten(), ypred.flatten())


# In[155]:

pickle.dump(ypred, open("CNN_Classify_%s.pkl" % seas, 'w'))


# # Train Regression

# In[156]:

precip_rows = (Y[train_rows] > 10.).mean(axis=1) > 0.10

Ytrain_reg = Ytrain[precip_rows]
Xtrain_reg = Xtrain[precip_rows]

cnn_reg = CNNDownscale(sess=tf.InteractiveSession())
cnn_reg.fit(Xtrain, Ytrain)
cnn_reg.predict(Xtrain)


# In[157]:

yclass = cnn_classify.predict(Xtest)
ypred = cnn_reg.predict(Xtest)

yhat = scale_y.inverse_transform(ypred) / 10
ytest_true = scale_y.inverse_transform(Ytest) / 10

yhat *= yclass > 0.5

from scipy.stats import pearsonr, spearmanr

print "Test Mean %f, Std: %f" % (yhat.mean(), ytest_true.std())
print "Prediction Mean %f, Std: %f" % (yhat.mean(), ytest_true.std())
print "Yhat shape", yhat.shape, "Ytest shape", Ytest.shape

print spearmanr(yhat.flatten(), ytest_true.flatten())
print "Pearson", pearsonr(yhat.flatten(), ytest_true.flatten())
print "RMSE:", np.mean((yhat - ytest_true)**2)**(0.5)

plt.figure(figsize=(12, 8))
plt.scatter(ytest_true, yhat, alpha=0.01)
plt.xlim([0, 140])
plt.ylim([0, 140])




# In[158]:

import scipy
def qqplot(x, y, ax=None, **kwargs):
    _, xr = scipy.stats.probplot(x, fit=False)
    _, yr = scipy.stats.probplot(y, fit=False)
    if ax is None:
        plt.scatter(xr, yr, **kwargs)
    else:
        ax.scatter(xr, yr, **kwargs)
        
        
plt.figure(figsize=(12,8))
qqplot(yhat.flatten(), ytest_true.flatten())
plt.ylim([0, 200])
plt.xlim([0, 200])
plt.xlabel("Projected")
plt.ylabel("Observed")


# # To Netcdf

# In[111]:

dfdata = []
print ypred.shape, Ytest.shape, locs.shape, t.shape
for j, (lat, lon) in enumerate(locs.values):
    for i, t1 in enumerate(ttest.values):
        d = dict(lat=lat, lon=lon, time=t1, projected=yhat[i,j], ground_truth=YtestTrue[i,j])
        dfdata.append(d)
df = pd.DataFrame(dfdata)
df.set_index(["lat", "lon", "time"], inplace=True)
dfx = xr.Dataset.from_dataframe(df)
dfx


# In[112]:

dfx['error'] = dfx.projected - dfx.ground_truth
print dfx
dfx.to_netcdf("/Users/tj/repos/pydownscale/scripts/results-data/%s/CNN_D_%s.nc" % (seas, seas))




# In[ ]:



