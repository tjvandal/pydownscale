from downscale import DownscaleModel, ASD, ASDMultitask
from MSSL import pMSSL
from sklearn.linear_model import LinearRegression, LassoCV, MultiTaskLasso
import numpy
import pickle
from data import DownscaleData
from sklearn import preprocessing

def asd_multitasklasso():
    model = MultiTaskLasso()
    f = "/home/vandal.t/repos/pydownscale/pydownscale/test_data/testdata.pkl"
    data = pickle.load(open(f, 'r'))
    asdm = ASDMultitask(data, model, season='JJA')
    asdm.train()
    out = asdm.predict(test_set=False)
    out.to_netcdf("test_data/mtl_test.nc")

def asd_mssl():
    model = pMSSL(max_epochs=5, quiet=False, lambd=1e10, gamma=1e-10, walgo='multiprocessor',
                  num_proc=48, w_epochs=50, omega_epochs=50)
    f = "/home/vandal.t/repos/pydownscale/pydownscale/test_data/testdata.pkl"
    lf = '/gss_gpfs_scratch/vandal.t/experiments/DownscaleData/newengland_MS_420_8835.pkl'
    data = pickle.load(open(lf, 'r'))
    asdm = ASDMultitask(data, model, season='JJA', ytransform=preprocessing.StandardScaler())
    asdm.train()
    out = asdm.predict(test_set=False)
    out.to_netcdf("test_data/ne_test_set.nc")

if __name__ == "__main__":
    asd_mssl()
