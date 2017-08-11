from base.sheet import *
from base.helpers import *
from models.abstract_application import App, AppHelper
from models.distributions import Distribution
from models.regression_models import HillMartyModel, ExtendedHillMartyModel
from models.performance_models import PerformanceModel
from models.risk_functions import RiskFunctionCollection
from models.uncertainty_models import UncertaintyModel
from utils.preprocessing import FileHelper
from utils.plotting import PlotHelper
from utils.kde import KDE, Transformations

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
    def __init__(self, design, perf, risk):
        self.design = design
        self.perf = perf
        self.risk = risk

args = None

def plot(d2perf, perf_model):
    d2mean = perf_model.get_mean(d2perf)
    d2var = perf_model.get_std(d2perf)
    logging.info('Reference perf: {}'.format(max(d2mean.values())))
    K = 200
    top_k = sorted(d2mean, key=lambda k: d2mean[k], reverse=True)[:K]
    top_perfs = {k:d2perf[k]._mcpts for k in top_k}
    top_perf = {k: d2perf[k] for k in top_k}
    d2risk = perf_model.get_risk(max(d2mean.values()), top_perf)
    for k in top_k:
        logging.debug('{}: {} {}, failure prob: {}'.format(
            k, d2mean[k], d2risk[k], d2perf[k] < kZeroThreshold))
    PlotHelper.plot_hists(top_perfs)
    points = [PRPoint(k, d2mean[k], d2risk[k]) for k in top_k]
    PlotHelper.plot_scatter({args.math_model: points}, annotate=True)

def load_from_file(perf_model):
    with open(args.load_path, 'r') as f:
        logging.debug('Loading from {}'.format(args.load_path))
        d2perf = pickle.load(f)
        f.close()
    plot(d2perf, perf_model)

def main():
    parse_args()
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)

    design_risk_scale = .1

    risk_func = RiskFunctionCollection.risk_function_collection[args.risk_func]
    perf_model = PerformanceModel(args.math_model, risk_func) 

    if args.load:
        load_from_file(perf_model)

    #===============
    # Example Usage
    #===============
    else:
        mean_f, mean_c = args.f, args.c
        assert args.apply_design and (not args.sigma1 is None) and (not args.sigma2 is None)
        series = defaultdict(list)
        candidate = tuple(args.apply_design)

        # Default values.
        perf_func = lambda x: math.sqrt(x)
        num_func = lambda x, y: y/x
        app = App('fake', mean_f, mean_c)

        # Let the performance reference bar be a dual-core processor.
        const_candidate = (0, 0, 0, 4, 0, 0)
        perf_model.add_index_bounds('i', upper=len(PerformanceModel.designs)+1)
        perf_model.compute_core_perfs(app)
        perf_model.add_target(perf_model.perf_target)
        const_perf = 1. * perf_model.apply_design(const_candidate, app)
        
        sigma1 = args.sigma1
        sigma2 = args.sigma2

        if args.fc_file and args.cpudb_dir:
            args.sigma1 = 0
            args.sigma2 = 0
            # fit f and c
            regress_model = ExtendedHillMartyModel()
            apps = AppHelper.gen_app(args.fc_file, regress_model)
            fs = [app.f for app in apps]
            f_pdf = KDE.fit(fs, Transformations.logit)
            sample_func_f = f_pdf.resample
            f_dist = CustomDistribution.EmpiricalDistribution(
                    sample_func_f, Transformations.sigmoid)
            logging.debug('Distribution of f: ({}, {})'.format(
                f_dist.mean, np.sqrt(f_dist.var)))
            c_dist = 0
            if isinstance(regress_model, ExtendedHillMartyModel):
                cs = [app.c for app in apps]
                c_pdf = KDE.fit(cs, np.log)
                sample_func_c = c_pdf.resample
                c_dist = CustomDistribution.EmpiricalDistribution(sample_func_c, np.exp)
                logging.debug('Distribution of c: ({}, {})'.format(
                    c_dist.mean, np.sqrt(c_dist.var)))
            app = App('Average', f_dist, c_dist)
                    
            # fit perf
            R2P = FileHelper.read_data_from_cpudb(args.cpudb_dir)
            sorted_R2P = sorted(R2P, key = lambda x: x[0])
            transistors = [p[0] for p in R2P]
            specint06 = [p[1] for p in R2P]
            logging.debug('Sample # for perf regression: {}'.format(len(transistors)))
            gpr = GPR(optimizer = 'fmin_l_bfgs_b', alpha = 10,
                    n_restarts_optimizer = 9, normalize_y = True)
            X = np.log(np.array(transistors)).reshape([len(transistors), 1])
            Y = np.log(np.array(specint06)).reshape([len(specint06), 1])
            trans_func = np.exp
            gpr.fit(X, Y)
            perf_func = lambda x: CustomDistribution.EmpiricalDistribution(
                    functools.partial(gpr.sample_y, np.log(x)), trans_func)
            num_func = fab_risk

        logging.info('sigma1, sigma2: {}, {}'.format(args.sigma1, args.sigma2))
        UncertaintyModel.set_rates(design_risk_scale * sigma2, sigma2)
        if not args.trans:
            if args.sigma1 > .0:
                app.set_f(Distribution.NormalizedBinomialDistribution(
                    mean_f, sigma1 * (1 - mean_f)))
                app.set_c(Distribution.NormalizedBinomialDistribution(
                    mean_c, sigma1 * mean_c))
        else:
            if args.sigma1 > .0:
                gt_f = Distribution.NormalizedBinomialDistribution(
                        mean_f, sigma1 * (1-mean_f))
                app.set_f(Distribution.BoxCoxTransformedDistribution(
                    mean_f, sigma1 * (1-mean_f), gt_f._mcpts, lower=0, upper=1))
                gt_c = Distribution.NormalizedBinomialDistribution(
                        mean_c, sigma1 * mean_c)
                app.set_c(Distribution.BoxCoxTransformedDistribution(
                    mean_c, sigma1 * mean_c, gt_c._mcpts, lower=0))
        perf_model.add_index_bounds('i', upper=len(PerformanceModel.designs)+1)
        perf_model.compute_core_perfs(app)
        perf_model.add_target(PerformanceModel.perf_target)
        d2perf = {candidate: perf_model.apply_design(candidate, app)}
        logging.info('Hist for {}, app: {}, sigma: {} {}, sample size: {}'.format(
            candidate, (app.f, app.c), sigma1, sigma2, len(d2perf[candidate]._mcpts)))
        PlotHelper.plot_hist(d2perf[candidate]._mcpts/const_perf)
        title = 'hist_f_' + str(args.f) + '_c_' + str(args.c)
        if args.trans:
            title = title + '_trans'
        PlotHelper.plot_hist(d2perf[candidate]._mcpts/args.bar, 'blue', title)
        d2mean = perf_model.get_mean(d2perf)
        d2risk = perf_model.get_risk(args.bar, d2perf)
        logging.info('average perf: {} ({})\narchitectural risk: {}'.format(
            d2mean[candidate]/args.bar, d2mean[candidate], d2risk[candidate]))
        #series[(mean_f, mean_c)].append(
        #        (sigma, d2mean[candidate]/const_perf, d2risk[candidate]))


def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel',
            help='set log level.')
    parser.add_argument('--path', action='store', dest='path',
            help='use regression model to get f, c.')
    parser.add_argument('--math_model', action='store', default='symmetric',
            help='which math model to use: symmetric, asymmetric or dynamic')
    parser.add_argument('--risk_func', action='store', dest='risk_func', default='linear',
            help='select risk model to use: step, linear or quad.')
    parser.add_argument('--analytical', action='store_true', dest='analytical', default=False,
            help='Use analytical approximation instead of MC.')
    parser.add_argument('--f', action='store', type=float, default=.0,
            help='Fixed f value to use.')
    parser.add_argument('--c', action='store', type=float, default=.0,
            help='Fixed c value to use.')
    parser.add_argument('--fc-file', action='store', dest='fc_file', default=None,
            help='filepath to empirical workload for fc regression.')
    parser.add_argument('--trans', action='store_true', default=False,
            help='Use boxcox transformation.')
    parser.add_argument('--cpudb-dir', action='store', dest='cpudb_dir', default=None,
            help='path to cpudb directory.')
    parser.add_argument('--sigma1', action='store', type=float, default=0,
            help='Application uncertainty level.')
    parser.add_argument('--sigma2', action='store', type=float, default=0,
            help='Architecture uncertainty level')
    parser.add_argument('--pickle', action='store', default='d2perf',
            help='File path to dump result.')
    parser.add_argument('--load', action='store_true', default=False,
            help='Use previouse results.')
    parser.add_argument('--load-path', action='store', default='d2perf',
            help='File path to load result.')
    parser.add_argument('--apply-design', type=int, nargs=len(PerformanceModel.designs), action='store',
            help='Apply a certain design.')
    parser.add_argument('--bar', type=float, default=0,
            help='Performance bar')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
