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

def Convert2PRPoint(xy, perf_model, d2perf, ref=None, ref_design=None, pareto_only=True):
    """ At a fixed uncertainty level.
    """

    logging.debug('Converting at {}'.format(xy))
    d2mean = perf_model.get_mean(d2perf)
    sorted_mean = sorted(d2mean.items(), key=lambda (k, v):v, reverse=True)
    max_mean = sorted_mean[0][1]
    best_design = sorted_mean[0][0]
    ref = ref if ref else max_mean
    d2risk = perf_model.get_risk(ref, d2perf)
    max_risk = d2risk[best_design] if d2risk[best_design] > 0 else 1.
    prpts = []
    for d in d2perf:
        if pareto_only:
            pareto = True
            for other in d2perf:
                if d2mean[other] > d2mean[d] and d2risk[other] < d2risk[d]:
                    pareto = False
                    break
            if pareto:
                prpts.append(PRPoint(d2mean[d]/ref, d2risk[d]/max_risk, d))
        else:
            if d2mean[d] >= 0.89 * ref:
                prpts.append(PRPoint(d2mean[d]/ref, d2risk[d]/max_risk, d))
    assert prpts
    return sorted(prpts, key=lambda k:k.perf)

def selected(k, select):
    selected = False
    for r in select:
        if abs(r[0] - k[0]) < 1e-5 and abs(r[1] - k[1]) < 1e-5:
            selected = True
    return selected

def prepare_plot(perf_model, xy2z):
    xy2perf = xy2z
    d2certain_perf = xy2perf[(0, 0)]
    d2certain_perf = sorted(d2certain_perf.items(), key=lambda (k,v): v, reverse=True)
    ref = d2certain_perf[0][1]
    ref_design = d2certain_perf[0][0]
    logging.info('Reference perf: {}'.format(ref))
    #select = [(0.8, 0.2), (1.0, 0.0), (0.8, 0.0), (1.0, 0.8), (0.4, 0.2), (0.2, 0.2)]
    select = [(0.8, 0.2)]
    xy2p = {k: Convert2PRPoint(k, perf_model, xy2z[k], ref, ref_design) for k in xy2z if selected(k, select)}
    xy2np = {k: Convert2PRPoint(k, perf_model, xy2z[k], ref, ref_design, False) for k in xy2z if selected(k, select)} if args.non_pareto else None
    #xy2p = {k: Convert2PRPoint(k, perf_model, xy2z[k], ref) for k in xy2z}
    return xy2p, xy2np

def plot_perf_risk_tradeoff(xy2pl, xy2sp=None):
    PlotHelper.plot_scatter(xy2pl, scatter=xy2sp, annotate=True)

def load_from_file():
    logging.info('Loading from {}'.format(args.path))
    with open(args.path, 'r') as f:
        data = pickle.load(f)
        f.close()
    logging.info('Loaded.')
    return data

def main():
    parse_args()
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    perf_model = PerformanceModel()

    data = load_from_file()
    data, data_all = prepare_plot(perf_model, data)
    plot_perf_risk_tradeoff(data, None if not args.non_pareto else data_all)
            
def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel',
            help='set log level.')
    parser.add_argument('--non-pareto', action='store_true', default=False,
            help='also plot non pareto points.')
    parser.add_argument('path', action='store',
            help='path to d2perf file.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
