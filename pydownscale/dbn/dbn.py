import sys
sys.path.append("/home/vandal.t/anaconda2/lib/python2.7/site-packages")
from pylearn2.config import yaml_parse
from pydownscale.data import DownscaleData
import os
import load_data
load_data.load_supervised  # how does this change anything for the mlp_layer ? 


seasons = ['DJF', 'MAM', 'JJA', 'SON']
season='DJF'

h1=500
h2=100
h3=50

mlp_yaml = "yamls/mlp_dropout.yaml"

if not os.path.exists("models/grbm_l1_%i_%s.pkl" % (h1, season)):
    layer1_yaml = open('yamls/grbm_l1.yaml', 'r').read() % dict(season=season, h1=h1)
    train = yaml_parse.load(layer1_yaml)
    train.main_loop()


if not os.path.exists("models/rbm_l2_%i_%i_%s.pkl" % (h1, h2, season)):
    layer2_yaml = open('yamls/rbm_l2.yaml', 'r').read() % dict(season=season, h1=h1, h2=h2)
    train = yaml_parse.load(layer2_yaml)
    train.main_loop()


if not os.path.exists("models/rbm_l3_%i_%i_%s.pkl" % (h2, h3, season)):
    layer3_yaml = open('yamls/rbm_l3.yaml', 'r').read() % dict(season=season, h1=h1, h2=h2, h3=h3)
    train = yaml_parse.load(layer3_yaml)
    train.main_loop()


mpl_pkl = "models/DBN_D_%s.pkl" % season
mpl_layer = open(mlp_yaml, 'r').read() % dict(season=season, h1=h1, h2=h2, h3=h3)
train = yaml_parse.load(mpl_layer)
train.main_loop()



