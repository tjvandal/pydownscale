from data import DownscaleData, read_nc_files, assert_bounds
import numpy
from downscale import DownscaleModel
from scipy.stats import pearsonr, spearmanr, kendalltau
import config
import pandas
import pickle 
import time

class DownscaleModelMPI:
    def __init__(self, data, model, training_size=config.train_percent, season=None, feature_columns=None):
        self.data = data
        self.model = model
        self.seasons =['DJF', 'MAM', 'JJA', 'SON']
        self.training_size = training_size


    # include quantile mapping ability
    def train(self, mpi_comm, mpi_ranks, lat='latitude', lon='longitude'):
        rank0 = mpi_ranks[0]
        curr_rank = mpi_comm.Get_rank()
        locations = None
        locations = numpy.array(self.data.location_pairs(lat, lon))
        params = [[lt, ln, seas] for lt, ln in locations for seas in self.seasons]
        params = numpy.array_split(params, len(mpi_ranks))
        myidx =  mpi_ranks.index(curr_rank)
        myparams = params[myidx]
        mymodels = []
        for lt, ln, seas in myparams:
            lt = float(lt)
            ln = float(ln)
            print "Rank: %i, Season: %s, Lat: %f, Lon: %f" % (curr_rank, seas, lt, ln)
            curr_model = DownscaleModel(self.data, self.model,season=seas,
                                        training_size=self.training_size,
                                        feature_columns=None) 
            curr_model.train(location={lat: lt, lon: ln})
            mymodels.append({lat: lt, lon: ln, 'season': seas, 'dmodel': curr_model})

        allmodels = mpi_comm.gather(mymodels, root=0)
        if curr_rank == 0: 
            self.models = pandas.DataFrame([m for l in allmodels for m in l])
            print self.models
        else:
            del allmodels

    def project_gcm(self, gcmdata):
        for seas in self.season:
            X = gcmdata.get_X(season=seas)
