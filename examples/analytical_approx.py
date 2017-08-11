from base.sheet import *
from base.helpers import *
from models.abstract_application import App, AppHelper
from models.distributions import Distribution
from models.math_models import MathModel
from models.regression_models import HillMartyModel, ExtendedHillMartyModel
from models.performance_models import PerformanceModel
from models.risk_functions import RiskFunctionCollection
from models.uncertainty_models import UncertaintyModel
from utils.preprocessing import FileHelper
from utils.plotting import PlotHelper
from uncertainties import ufloat
from mcerp import N

from collections import defaultdict
import argparse
import functools
import math
import matplotlib
import logging
import itertools
import pickle

args = None

def load_from_file(perf_model):
    with open(args.load_path, 'r') as f:
        logging.debug('Loading from {}'.format(args.load_path))
        d2perf = pickle.load(f)
        f.close()
    return d2perf

def main():
    parse_args()
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)

    #===============
    # Example Usage
    #===============
    risk_func = RiskFunctionCollection.risk_function_collection[args.risk_func]
    logging.info('Using risk function: {}'.format(risk_func.get_name()))
    perf_model = PerformanceModel(args.math_model, risk_func, args.analytical)
    UncertaintyModel.set_rates(0.001, 0.01)
    perf_func = UncertaintyModel.core_perf_boxcox()

    if args.load:
        d2perf = load_from_file(perf_model)
    else:
        app = App('fake', args.f, args.c)
        d2perf = perf_model.get_perf(app)
        with open(args.save_path+'_' +str(args.f)+'_'+str(args.c), 'w') as f:
            pickle.dump(d2perf, f)
            f.close()
    logging.info('d2perf: {}'.format(d2perf))

def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel', default='info',
            help='set log level.')
    parser.add_argument('--path', action='store',
            help='use regression model to get f, c.')
    parser.add_argument('--math_model', action='store',
            help='which math model to use: symmetric, asymmetric, dynamic or hete')
    parser.add_argument('--risk_func', action='store', default='linear',
            help='select risk model to use: step, linear or quad.')
    parser.add_argument('--analytical', action='store_true', default=False,
            help='Use analytical approximation instead of MC.')
    parser.add_argument('--f', action='store', type=float, default=.9,
            help='Fixed f value to use.')
    parser.add_argument('--c', action='store', type=float, default=.0,
            help='Fixed c value to use.')
    parser.add_argument('--save-path', action='store', default='d2perf',
            help='File path to dump result.')
    parser.add_argument('--load', action='store_true', default=False,
            help='Use previouse results.')
    parser.add_argument('--load-path', action='store', default='d2perf',
            help='File path to load result.')
    parser.add_argument('--eval-design', type=int, nargs=len(PerformanceModel.designs), action='store',
            help='Evaluate a given design.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
