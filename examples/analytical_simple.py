from base.sheet import *
from base.helpers import *
from models.abstract_application import App, AppHelper
from models.custom_distributions import CustomDistribution
from models.regression_models import HillMartyModel, ExtendedHillMartyModel
from models.performance_models import MathModel
from models.risk_functions import RiskFunctionCollection
from models.uncertainty_models import UncertaintyModel
from utils.preprocessing import FileHelper
from utils.plotting import PlotHelper
from uncertainties import ufloat
from uncertainties.core import AffineScalarFunc
from mcerp import N

from collections import defaultdict
import argparse
import functools
import math
import matplotlib
import logging
import itertools
import pickle

class PerformanceModel(MathModel):
    area = 256
    perf_target = 'speedup'
    #designs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    designs = [8, 16, 32, 64, 128, 256]
    def __init__(self, selected_model, risk_function, analytical):
        self.sheet1 = Sheet(analytical)
        self.sheet1.addSyms(MathModel.config_syms +
                MathModel.perf_syms +
                MathModel.stat_syms +
                MathModel.power_syms)

        if selected_model == 'symmetric':
            model = MathModel.symm_exprs
        elif selected_model == 'asymmetric':
            model = MathModel.asymm_exprs
        elif selected_model == 'dynamic':
            model = MathModel.dynamic_exprs
        elif selected_model == 'hete':
            model = MathModel.hete_exprs
        else:
            raise ValueError('Unrecgonized model: {}'.format(selected_model))

        self.given = defaultdict()
        self.ims = defaultdict()
        self.add_given('area_total', self.__class__.area)
        self.target = [self.__class__.perf_target]
        self.sheet1.addExprs(MathModel.common_exprs + model)
        self.sheet1.addPreds(response=self.target)
        self.risk_func = risk_function

    def gen_feed(self, k2v):
        feed = []
        for k, v in k2v.iteritems():
            feed += [(k, v)]
        return feed

    def compute(self, app):
        logging.debug('Solving app: {}'.format(app.get_printable()))
        given = self.gen_feed(self.given) + app.gen_feed()
        ims = self.gen_feed(self.ims)
        logging.debug('-- Given: {} -- Intermediates: {}'.format(
            given, {k: v.func.__name__ for k, v in self.ims.iteritems()}))
        self.sheet1.addPreds(given=given, intermediates=ims)
        result = self.sheet1.compute() # Return a map of all target results.
        return result

    def dump(self):
        # Dumping all expressions.
        self.sheet1.dump()
        # Dumping risk function used.
        print self.risk_func

    def add_given(self, name, val):
        self.given[name] = val

    def add_uncertain(self, name, partial_func):
        self.ims[name] = partial_func

    def get_risk(self, ref, d2perf):
        """ Computes risk for d2perf w.r.t. ref

        Args:
            ref: reference performance bar
            d2perf: performance array-like

        Returns:
            single float (mean risk)
        """
        return {k:self.risk_func.get_risk(ref, v) for k, v in d2perf.iteritems()}

    def get_mean(self, d2uc):
        """ Extracts mean performance.
        """
        return {k:self.get_numerical(v) for k, v in d2uc.iteritems()}

    def get_std(self, d2uc):
        return {k:math.sqrt(self.get_var(v)) for k, v in d2uc.iteritems()}

    def get_perf(self, app):
        """ Computes peformance distribution over a single app.
        """
        d2perf = defaultdict()

        # classic hill&marty model
        for d in self.__class__.designs:
            self.add_given('small_core_size', d)
            result = self.compute(app)
            perf = result[self.perf_target]
            #if isinstance(perf, AffineScalarFunc):
            #    perf = N(perf.n, (perf.std_dev) ** 2)
            d2perf[d] = perf

        return d2perf

    def print_latex(self):
        self.sheet1.printLatex()

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
    if args.analytical:
        logging.info('Performing analytical error propagation.')
    perf_model = PerformanceModel(args.math_model, risk_func, args.analytical)
    perf_func = UncertaintyModel.core_performance_uncertainty(.1)
    perf_model.add_uncertain('small_core_perf', perf_func)
    #perf_model.dump()

    if args.load:
        d2perf = load_from_file(perf_model)
    else:
        app = App('fake', args.f, args.c)
        d2perf = perf_model.get_perf(app)
        #d2perfs = {d: perf._mcpts for d, perf in d2perf.iteritems()}
        #PlotHelper.plot_hists(d2perfs)
        with open(args.pickle+'_' +str(args.f)+'_'+str(args.c), 'w') as f:
            pickle.dump(d2perf, f)
            f.close()
    logging.info('d2perf: {}'.format(d2perf))

def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel',
            help='set log level.')
    parser.add_argument('--path', action='store', dest='path',
            help='use regression model to get f, c.')
    parser.add_argument('--math_model', action='store', default='symmetric',
            help='which math model to use: symmetric, asymmetric, dynamic or hete')
    parser.add_argument('--risk_func', action='store', dest='risk_func', default='linear',
            help='select risk model to use: step, linear or quad.')
    parser.add_argument('--analytical', action='store_true', dest='analytical', default=False,
            help='Use analytical approximation instead of MC.')
    parser.add_argument('--f', action='store', type=float, default=.0,
            help='Fixed f value to use.')
    parser.add_argument('--c', action='store', type=float, default=.0,
            help='Fixed c value to use.')
    parser.add_argument('--pickle', action='store', default='d2perf',
            help='File path to dump result.')
    parser.add_argument('--load', action='store_true', default=False,
            help='Use previouse results.')
    parser.add_argument('--load-path', action='store', default='d2perf',
            help='File path to load result.')
    parser.add_argument('--apply-design', type=int, nargs=len(PerformanceModel.designs), action='store',
            help='Apply a certain design.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
