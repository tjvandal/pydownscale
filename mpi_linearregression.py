from mpi4py import MPI
import os
import sys
import pandas
import numpy
from sklearn.linear_model import LinearRegression

pydownscale_path = os.path.dirname(os.path.dirname(__file__))
sys.path.append(pydownscale_path)
from pydownscale.data import DownscaleData, read_nc_files
from pydownscale.downscale import DownscaleModel
import argparse

parser = argparse.ArgumentParser(description='Datafiles.')
parser.add_argument('--cmip5_dir', dest='cmip5_dir')
parser.add_argument('--cpc_dir', dest='cpc_dir')
args = parser.parse_args()


comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

# initialize downscale data
cmip5_dir = args.cmip5_dir   #/Users/tj/data/cmip5/access1-0/"
cpc_dir = args.cpc_dir      # /Users/tj/data/usa_cpc_nc/merged"

# climate model data, monthly
cmip5 = read_nc_files(cmip5_dir)
cmip5.time = pandas.to_datetime(cmip5.  time.values)
cmip5 = cmip5.resample('MS', 'time', how='mean')   ## try to not resample

# daily data to monthly
cpc = read_nc_files(cpc_dir)
cpc.time = pandas.to_datetime(cpc.time.values)
monthlycpc = cpc.resample('MS', dim='time', how='mean')  ## try to not resample

data = DownscaleData(cmip5, monthlycpc)

if rank == 0:
   ## lets split up our y's
   pairs = data.location_pairs('lat', 'lon')[:size]
   print "Number of pairs:", len(data.location_pairs('lat', 'lon'))
else:
   pairs = None

pairs = comm.scatter(pairs, root=0)
linearmodel = LinearRegression()
dmodel = DownscaleModel(data, linearmodel)
dmodel.train(location={'lat': pairs[0], 'lon': pairs[1]})
newData = comm.gather(dmodel.get_results(), root=0)

if rank == 0:
   print 'master:', newData
