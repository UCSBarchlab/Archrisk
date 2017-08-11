from base.sheet import *
from base.helpers import *
from models.abstract_application import App, AppHelper
from models.distributions import Distribution
from models.math_models import MathModel
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

args = None

def plot_decision_shift(xy2z, perf_model):
    # xy2z is actuall (x, y)(design) -> perf, so xy2z[x, y][d] -> perf.
    if (0, 0) not in xy2z:
        logging.info('(sigma1, sigma2): ({}, {})'.format(sigma1, sigma2))
        UncertaintyModel.set_rates(0, 0)
        app = App('fake', args.f, args.c)
        d2const_perf = perf_model.get_perf(app)
        xy2z[(0, 0)] = d2const_perf
    xy2perf = xy2z
    xy2mean = {k: perf_model.get_mean(xy2perf[k]) for k in xy2perf}
    xy2std = {k: perf_model.get_std(xy2perf[k]) for k in xy2perf}
    xy2sorted = {coord: sorted(xy2mean[coord].items(),
        key=lambda (k, v): v, reverse=True) for coord in xy2mean}
    xy2best_mean = {coord: xy2sorted[coord][0][1] for coord in xy2sorted}
    xy2risk = {k: perf_model.get_risk(xy2best_mean[k], xy2perf[k]) for k in xy2perf}
    xy2risk_sorted = {coord: sorted(xy2risk[coord].items(),
        key=lambda (k, v): v) for coord in xy2risk}
    xy2best = {coord: xy2sorted[coord][0] for coord in xy2sorted}
    xy2best_design = {coord: xy2sorted[coord][0][0] for coord in xy2sorted}
    xy2risk_best_design = {coord: xy2risk_sorted[coord][0][0] for coord in xy2risk_sorted}
    xy2best_risk = {coord: xy2risk_sorted[coord][0][1] for coord in xy2risk_sorted}

    #for coord in xy2perf:
    #    assert(xy2sorted[coord][0][1] != xy2sorted[coord][1][1])
    #    assert(xy2risk_sorted[coord][0][1] != xy2risk_sorted[coord][1][1])
    #assert(xy2best_design[(0, 0)] == xy2risk_best_design[(0, 0)])
    #logging.warn('best certain perf design not equal best certain risk.')
    
    # Plot top K points.
    #K = 20
    #top_k = sorted_d2mean[:K]
    #top_perfs = {k:d2perf[k]._mcpts for k in top_k}
    #top_perf = {k: d2perf[k] for k in top_k}
    #d2risk = perf_model.get_risk(max(d2mean.values()), top_perf)
    #for k in top_k:
    #    logging.debug('{}: {} {}, failure prob: {}'.format(
    #        k, d2mean[k], d2risk[k], d2perf[k] < kZeroThreshold))
    #PlotHelper.plot_hists(top_perfs)

    # Compute prob.
    #points = [(coord, 
    #    xy2perf[coord][xy2best_design[coord]] - xy2perf[coord][xy2best_design[(0, 0)]] > .0) for coord in xy2best_design if coord != (0, 0)]
    #PlotHelper.plot_shift_prob(points)

    # Log results.
    for coord in xy2perf:
        logging.info(('{}:\nBest Certain Design: {} ({}) ({}, {})\n' +
            'Perf Opt Design: {} ({}) ({} <- {}, {})\n' +
            'Risk Opt Design: {} ({}) ({}, {} <- {})\n' +
            'Best Possible Performance: {}').format(
            coord,
            xy2best_design[(0, 0)], Metrics.h_metric(xy2best_design[(0, 0)]), 
            xy2mean[coord][xy2best_design[(0, 0)]], xy2risk[coord][xy2best_design[(0, 0)]], 
            xy2best_design[coord], Metrics.h_metric(xy2best_design[coord]),
            xy2mean[coord][xy2best_design[coord]], xy2mean[coord][xy2best_design[(0, 0)]], 
            xy2risk[coord][xy2best_design[coord]],
            xy2risk_best_design[coord], Metrics.h_metric(xy2risk_best_design[coord]),
            xy2mean[coord][xy2risk_best_design[coord]], xy2risk[coord][xy2risk_best_design[coord]],
            xy2risk[coord][xy2best_design[(0, 0)]],
            xy2best_mean[coord]))

    # Compute equality for ploting shift figure.
    points = [(coord,
        xy2best_design[coord] == xy2best_design[(0, 0)],
        xy2risk_best_design[coord] == xy2best_design[(0, 0)],
        xy2best_design[coord] == xy2risk_best_design[coord],
        ) for coord in xy2best_design]

    if not args.fc_file and not args.cpudb_dir:
        title = 'design_shift_' + str(args.f) + '_' + str(args.c)
    elif not args.fc_file:
        title = 'cpudb_design_shift_' + str(args.f) + '_' + str(args.c)
    elif not args.cpudb_dir:
        title = ('kde_design_shift_' +
                args.fc_file[args.fc_file.rfind('/')+1:args.fc_file.rfind('.')])
    else:
        title = ('cpudb_kde_design_shift_' +
                args.fc_file[args.fc_file.rfind('/')+1:args.fc_file.rfind('.')])
    PlotHelper.plot_shift_scatter(points, title)

def load_from_file(perf_model):
    with open(args.load_path, 'r') as f:
        logging.debug('Loading from {}'.format(args.load_path))
        xy2z = pickle.load(f)
        f.close()
    logging.debug('Loaded.')
    plot_decision_shift(xy2z, perf_model)

def main():
    parse_args()
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)

    # Design constants.
    design_risk_scale = .2

    risk_func = RiskFunctionCollection.risk_function_collection[args.risk_func]
    perf_model = PerformanceModel(args.math_model, risk_func) 

    if args.load:
        load_from_file(perf_model)

    else:
        # Set application type.
        mean_f, mean_c = args.f, args.c
        app = App('fake', mean_f, mean_c)

        xy2z = defaultdict()

        perf_func = lambda x: math.sqrt(x)
        num_func = lambda x, y: y/x

        sigmas = np.arange(0, 1., .2)
        sigma1 = sigmas[0]
        sigma2 = sigmas[0]

        # fc empirical, sigma2: design
        if args.fc_file:
            sigma1 = None
            regress_model = ExtendedHillMartyModel()
	    apps = AppHelper.gen_app(args.fc_file, regress_model)
            fs = np.asarray([app.f for app in apps])
            BoxCox.test(fs, la=20, lb=100)
            f_dist = Distribution.DistributionFromBoxCoxGaussian(.8, .2, fs)

            # KDE
            #f_pdf = KDE.fit(fs, Transformations.logit)
            #sample_func_f = f_pdf.resample
            #f_dist = Distribution.EmpiricalDistribution(
            #        sample_func_f, Transformations.sigmoid)
            #logging.debug('Distribution of f: ({}, {})'.format(f_dist.mean, np.sqrt(f_dist.var)))

            c_dist = 0
            if isinstance(regress_model, ExtendedHillMartyModel):
                cs = [app.c for app in apps]
                BoxCox.test(cs, la=-10, lb=10)
                c_dist = Distribution.DistributionFromBoxCoxGaussian(.2, .2, cs)

                # KDE
                #c_pdf = KDE.fit(cs, np.log)
                #sample_func_c = c_pdf.resample
                #c_dist = Distribution.EmpiricalDistribution(sample_func_c, np.exp)
                #logging.debug('Distribution of c: ({}, {})'.format(
                #    c_dist.mean, np.sqrt(c_dist.var)))

                # Plot the fitting in transform space.
                #fx = np.linspace(-1.5, 20, 100)
                #PlotHelper.plot_KDE(fx, f_pdf, None, [Transformations.logit(f) for f in fs])
                #cx = np.linspace(-5.5, -3.5, 100)
                #PlotHelper.plot_KDE(cx, c_pdf, None, [np.log(c) for c in cs])

            app = App('Average', f_dist, c_dist)

        if args.cpudb_dir:
            sigma2 = None
            R2P = FileHelper.read_data_from_cpudb(args.cpudb_dir)
            sorted_R2P = sorted(R2P, key = lambda x: x[0])
            #R2P = [[x, np.power(x, .37)] for x in np.random.uniform(0, 100, 100)]
            transistors = [p[0] for p in R2P]
            specint06 = [p[1] for p in R2P]
            logging.debug('Sample # for perf regression: {}'.format(len(transistors)))

            # Scipy multivariate KDE
               
            #x2y = dict(zip(transistors, specint06))
            #pdf = KDE.fit(np.array([transistors, specint06]))
            #samples = pdf.resample(400)
            #transistors = samples[0]
            #specint06 = samples[1]
            #PlotHelper.plot_trend(x2y)
            #PlotHelper.plot_trend(dict(zip(samples[0], samples[1])))

            # Lmfit pollack's rule + gaussian error
            #pmodel = PollackModel()
            #result = pmodel.fit(transistors, specint06)
            #pp = result['p']
            #shift = np.max(specint06)
            #diff = np.asarray([shift + perf - np.power(trans, pp)
            #    for perf, trans in zip(specint06, transistors)])
            #BoxCox.test(diff)

            # Scikit-learn GPR
            gpr = GPR(optimizer = 'fmin_l_bfgs_b', alpha = 10,
                    n_restarts_optimizer = 9, normalize_y = True)
            X = np.log(np.array(transistors)).reshape([len(transistors), 1])
            Y = np.log(np.array(specint06)).reshape([len(specint06), 1])
            trans_func = np.exp
            gpr.fit(X, Y)
            logging.debug('GP Quality LML (Higher is better): {}'.format(
                gpr.log_marginal_likelihood_value_))

            perf_func = lambda x: Distribution.EmpiricalDistribution(
                    functools.partial(gpr.sample_y, np.log(x)), trans_func)
            num_func = fab_risk

            # Sampling and plotting
            #sample_x = np.log(8)
            #sample_func = functools.partial(gpr.sample_y, sample_x)
            #perf_dist = Distribution.EmpiricalDistribution(sample_func, np.exp)
            #logging.debug('Resampled perf: ({}, {})'.format(
            #    perf_dist.mean, np.sqrt(perf_dist.var)))
            #x2y = dict(zip((transistors), (specint06)))
            #samples = [(np.exp(sample_x), y) for y in perf_dist._mcpts]
            #PlotHelper.plot_gp_regression(gpr.predict, np.exp, x2y, samples)

        # sigma1: fc, sigma2: design
        if sigma1 is None and sigma2 is None:
            raise NotImplementedError
            logging.info('Fixed design point!')
            d2uncert_perf = perf_model.get_perf(num_func, perf_func, app)
            sigma1, sigma2 = 0, 0
            xy2z[(sigma1, sigma2)] = d2uncert_perf

        elif sigma1 is None:
            for sigma2 in sigmas:
                logging.info('sigma2: {}'.format(sigma2))
                UncertaintyModel.set_rates(sigma2 * design_risk_scale, sigma2)
                d2uncert_perf = perf_model.get_perf(app)
                xy2z[(0, sigma2)] = d2uncert_perf

        elif sigma2 is None:
            raise NotImplementedError
            for sigma1 in sigmas:
                logging.info('sigma1: {}'.format(sigma1))
                app = App('fake', mean_f, mean_c)
                if sigma1 > .0:
                    app.set_f(Distribution.NormalizedBinomialDistribution(
                        mean_f, sigma1 * (1-mean_f)))
                    app.set_c(Distribution.NormalizedBinomialDistribution(
                        mean_c, sigma1 * mean_c))
                d2uncert_perf = perf_model.get_perf(num_func, perf_func, app)
                xy2z[(sigma1, 0)] = d2uncert_perf

        else:
            for sigma1 in sigmas:
                for sigma2 in sigmas:
                    logging.info('(sigma_app, sigma_arch): ({}, {})'.format(sigma1, sigma2))
                    # Setting the ucertain level.
                    UncertaintyModel.set_rates(sigma2*design_risk_scale, sigma2)
                    app = App('abs', mean_f, mean_c)
                    if not args.trans:
                        # Use groundtruth distributions.
                        if sigma1 > .0:
                            app.set_f(Distribution.NormalizedBinomialDistribution(
                                mean_f, sigma1 * (1-mean_f)))
                            app.set_c(Distribution.NormalizedBinomialDistribution(
                                mean_c, sigma1 * mean_c))
                    else:
                        # Use boxcox distributions.
                        if sigma1 > .0:
                            # Step 1: sample from groundtruth.
                            # Step 2: generate distribution from box-cox transformation.
                            gt_f = Distribution.NormalizedBinomialDistribution(
                                    mean_f, sigma1 * (1-mean_f))
                            f_dist = Distribution.DistributionFromBoxCoxGaussian(
                                    mean_f, sigma1 * (1-mean_f), gt_f._mcpts, lower=0, upper=1)
                            gt_c = Distribution.NormalizedBinomialDistribution(
                                    mean_c, sigma1 * mean_c)
                            c_dist = Distribution.DistributionFromBoxCoxGaussian(
                                    mean_c, sigma1 * mean_c, gt_c._mcpts, lower=0)
                            app = App('approx', f_dist, c_dist)
                    d2uncert_perf = perf_model.get_perf(app)
                    xy2z[(sigma1, sigma2)] = d2uncert_perf
            
        # Write results to file.
        with open(args.pickle+'_' +str(args.f)+'_'+str(args.c), 'w') as f:
            pickle.dump(xy2z, f)
            f.close()
            
        # Plot shifting figure.
        plot_decision_shift(xy2z, perf_model)

def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel',
            help='set log level.')
    parser.add_argument('--math-model', action='store', default='symmetric',
            choices=MathModel.names(),
            help='which math model to use: symmetric, asymmetric or dynamic')
    parser.add_argument('--risk-func', action='store', dest='risk_func',
            default='linear', choices=RiskFunctionCollection.risk_function_collection,
            help='select risk model to use: step, linear or quad.')
    parser.add_argument('--f', action='store', type=float, default=.9,
            help='Fixed f value to use.')
    parser.add_argument('--c', action='store', type=float, default=.01,
            help='Fixed c value to use.')
    parser.add_argument('--fc-file', action='store', dest='fc_file', default=None,
            help='filepath to empirical workload for fc regression.')
    parser.add_argument('--cpudb-dir', action='store', dest='cpudb_dir', default=None,
            help='path to cpudb directory.')
    parser.add_argument('--trans', action='store_true', dest='trans', default=False,
            help='use transformed Gaussian as inputs.')
    parser.add_argument('--use-power', action='store_true', dest='use_power', default=False,
            help='use power equations.')
    parser.add_argument('--pickle', action='store', default='d2p',
            help='File path to dump result.')
    parser.add_argument('--load', action='store_true', default=False,
            help='Use previouse results.')
    parser.add_argument('--load-path', action='store', default='d2p',
            help='File path to load result.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
