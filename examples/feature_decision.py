from base.sheet import *
from base.helpers import *
from models.abstract_application import App, AppHelper
from models.distributions import Distribution
from models.metrics import Metrics
from models.regression_models import HillMartyModel, ExtendedHillMartyModel, PollackModel
from models.performance_models import PerformanceModel
from models.risk_functions import RiskFunctionCollection
from models.uncertainty_models import UncertaintyModel
from utils.preprocessing import FileHelper
from utils.plotting import PlotHelper
from utils.kde import KDE, Transformations
from utils.boxcox import BoxCox

from collections import defaultdict
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF
import argparse
import functools
import math
import matplotlib
import logging
import itertools
import pickle

kZeroThreshold = 1e-2

class PRPoint(object):
    def __init__(self, perf, risk, design):
        self.perf = perf
        self.risk = risk
        self.design = design

args = None

def Convert2PRPoint(xy, perf_model, d2perf, ref=None):
    """ At a fixed uncertainty level.
    """

    logging.debug('Converting at {}'.format(xy))
    d2mean = perf_model.get_mean(d2perf)
    sorted_mean = sorted(d2mean.items(), key=lambda (k, v):v, reverse=True)
    max_mean = sorted_mean[0][1]
    best_design = sorted_mean[0][0]
    ref = ref if ref else max_mean
    d2risk = perf_model.get_risk(ref, d2perf)
    prpts = []
    for d in d2perf:
        pareto = True
        for other in d2perf:
            if d2mean[other] > d2mean[d] and d2risk[other] < d2risk[d]:
                pareto = False
                break
        if pareto:
            prpts.append(PRPoint(d2mean[d], d2risk[d], d))
    assert prpts
    return prpts

def prepare_plot(perf_model, xy2z, xy2z_new):
    xy2perf = xy2z
    new_xy2perf = xy2z_new

    d2certain_perf = xy2perf[(0, 0)]
    ref = max(d2certain_perf.values())

    xy2p = {k: Convert2PRPoint(k, perf_model, xy2z[k], ref) for k in xy2z}
    xy2p_new = {k: Convert2PRPoint(k, perf_model, xy2z_new[k], ref) for k in xy2z_new}
    data = {}
    for k in xy2p:
        pts = []
        d2p = xy2p[k]
        d2p_new = xy2p_new[k]
        for d in d2p:
            pareto = True
            for other in d2p_new:
                #TODO: get the pareto frontier from both settings.
        pareto = True
        if 
    return data

def plot_perf_risk_tradeoff(xy2p):
    PlotHelper.plot_scatter(xy2p)

def load_from_files():
    logging.info('Loading from {}'.format(args.path))
    with open(args.path, 'r') as f:
        data = pickle.load(f)
        f.close()
    logging.info('Loading from {}'.format(args.feature_path))
    with open(args.feature_path, 'r') as f:
        new_data = pickle.load(f)
        f.close()
    logging.info('Loaded.')
    return data, new_data

def main():
    parse_args()
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    perf_model = PerformanceModel()

    data, new_data = load_from_files()
    data = prepare_plot(perf_model, data, new_data)
    plot_perf_risk_(data)
            
def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel',
            help='set log level.')
    parser.add_argument('path', action='store',
            help='path to d2perf file.')
    parser.add_argument('feature_path', action='store',
            help='path to d2perf file with new design feature.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
