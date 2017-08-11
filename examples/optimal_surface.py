from base.sheet import *
from base.helpers import *
from models.abstract_application import App, AppHelper
from models.custom_distributions import CustomDistribution
from models.metrics import Metrics
from models.regression_models import HillMartyModel, ExtendedHillMartyModel, PollackModel
from models.performance_models import MathModel
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

class ShiftPoint(object):
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z

class PerformanceModel(MathModel):
    area = 256
    perf_target = 'speedup'
    # Too much to simulate with MC, need more efficient methods.
    #designs = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    designs = [8, 16, 32, 64, 128, 256]
    def __init__(self, selected_model, risk_function):
        self.sheet1 = Sheet()
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
        result = self.sheet1.compute() # Map of all target results.
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
        """ Compute risk for d2perf w.r.t. ref

        Args:
            ref: reference performance bar
            d2perf: performance array-like

        Returns:
            single float (mean risk)
        """
        return {k: self.risk_func.get_risk(ref, v) for k, v in d2perf.iteritems()}

    def get_mean(self, d2uc):
        """ Extracts mean performance.
        """
        return {k: self.get_numerical(v) for k, v in d2uc.iteritems()}

    def get_std(self, d2uc):
        return {k: math.sqrt(self.get_var(v)) for k, v in d2uc.iteritems()}

    def print_latex(self):
        self.sheet1.printLatex()

    def apply_candidate(self, candidate, num_func, perf_func, app):
        assert(len(candidate) == len(self.designs))
        area_left = self.__class__.area - sum([x * y for x, y in zip(candidate, self.designs)])
        logging.debug('Applying candidate: {} ({})'.format(candidate, area_left))
        working_core_num, score_perf, pcore_perf = 0, 0, 0
        candidate_ns, candidate_ps = [], []
        if area_left > 0:
            candidate_ns.append(num_func(area_left, area_left))
            candidate_ps.append(perf_func(area_left))
        for x, y in zip(candidate, self.designs):
            if x > 0:
                num_cur = num_func(y, x * y)
                working_core_num += num_cur
                candidate_ns.append(num_cur)
                perf_cur = perf_func(y)
                candidate_ps.append(perf_cur)
        score_perf = CustomDistribution.MaxBinomial(candidate_ns, candidate_ps)
        pcore_perf = sum([n * p for n, p in zip(candidate_ns, candidate_ps)])
        self.add_given('score_perf', score_perf)
        self.add_given('working_core_num', working_core_num)
        self.add_given('pcore_perf', pcore_perf)
        perf = self.compute(app)[self.perf_target]
        logging.debug('Applying result: perf {}'.format(perf))
        return {candidate: perf}

args = None

def evaluate_with_uncertainty(perf_model, sigma1, sigma2, candidate):
    """ Evaluate candidate perf under ground truth uncertainty level (sigma1, sigma2)
    """
    fab_d = .003
    fab_a = .5
    design_risk_scale = .1

    fab_risk = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)

    mean_f, mean_c = args.f, args.c
    logging.debug('Evaluating {} under uncertainty ({}, {})'.format(candidate, sigma1, sigma2))
    if args.fc_file:
        sigma1 = None
        regress_model = ExtendedHillMartyModel()
        apps = AppHelper.gen_app(args.fc_file, regress_model)
        fs = np.asarray([app.f for app in apps])
        BoxCox.test(fs, la=20, lb=100)
        transformed_data, lambda_ = BoxCox.transform(fs)
        f_dist = CustomDistribution.BoxCoxTransformedDistribution(.8, .2, lambda_)

        # KDE
        #f_pdf = KDE.fit(fs, Transformations.logit)
        #sample_func_f = f_pdf.resample
        #f_dist = CustomDistribution.EmpiricalDistribution(
        #        sample_func_f, Transformations.sigmoid)
        #logging.debug('Distribution of f: ({}, {})'.format(f_dist.mean, np.sqrt(f_dist.var)))

        c_dist = 0
        if isinstance(regress_model, ExtendedHillMartyModel):
            cs = [app.c for app in apps]
            BoxCox.test(cs, la=-10, lb=10)
            transformed_data, lambda_ = BoxCox.transform(cs)
            c_dist = CustomDistribution.BoxCoxTransformedDistribution(.2, .2, lambda_)

            # KDE
            #c_pdf = KDE.fit(cs, np.log)
            #sample_func_c = c_pdf.resample
            #c_dist = CustomDistribution.EmpiricalDistribution(sample_func_c, np.exp)
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

        perf_func = lambda x: CustomDistribution.EmpiricalDistribution(
                functools.partial(gpr.sample_y, np.log(x)), trans_func)
        num_func = fab_risk

        # Sampling and plotting
        #sample_x = np.log(8)
        #sample_func = functools.partial(gpr.sample_y, sample_x)
        #perf_dist = CustomDistribution.EmpiricalDistribution(sample_func, np.exp)
        #logging.debug('Resampled perf: ({}, {})'.format(
        #    perf_dist.mean, np.sqrt(perf_dist.var)))
        #x2y = dict(zip((transistors), (specint06)))
        #samples = [(np.exp(sample_x), y) for y in perf_dist._mcpts]
        #PlotHelper.plot_gp_regression(gpr.predict, np.exp, x2y, samples)


    # sigma1: fc, sigma2: design
    if sigma1 is None and sigma2 is None:
        logging.info('Empirical design point.')
    elif sigma1 is None:
        logging.info('sigma2: {}'.format(sigma2))
        perf_func = lambda x: math.sqrt(x)
        num_func = lambda x, y: y/x
        if sigma2 > .0:
            num_func = fab_risk
            #perf_func = UncertaintyModel.core_design_performance_uncertainty(
            #        sigma2 * design_risk_scale, sigma2)
            perf_func = UncertaintyModel.core_performance_uncertainty(sigma2)
    elif sigma2 is None:
        logging.info('sigma1: {}'.format(sigma1))
        app = App('fake', mean_f, mean_c)
        if sigma1 > .0:
            app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                mean_f, sigma1 * (1-mean_f)))
            app.set_c(CustomDistribution.NormalizedBinomialDistribution(
                mean_c, sigma1 * mean_c))
    else:
        logging.info('(sigma1, sigma2): ({}, {})'.format(sigma1, sigma2))
        if not args.trans:
            # We are using hand-written 'ground truth' distributions.
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            app = App('fake', mean_f, mean_c)
            if sigma1 > .0:
                app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                    mean_f, sigma1 * (1-mean_f)))
                app.set_c(CustomDistribution.NormalizedBinomialDistribution(
                    mean_c, sigma1 * mean_c))
            if sigma2 > .0:
                num_func = fab_risk
                perf_func = UncertaintyModel.core_design_performance_uncertainty(
                        sigma2 * design_risk_scale, sigma2)
                #perf_func = UncertaintyModel.core_performance_uncertainty(sigma2)
    d2perf = perf_model.apply_candidate(candidate, num_func, perf_func, app) 
    return d2perf

def get_xy2normed_perf(perf_model, xy2z, reevaluate=False):
    # xy2z is actually (x, y)(design) -> perf, so xy2z[x, y][d] -> perf.
    if reevaluate:
        logging.info('===Reevaluating===')
    assert (0, 0) in xy2z
    xy2perf = xy2z
    xy2mean = {k: perf_model.get_mean(xy2perf[k]) for k in xy2perf}
    xy2std = {k: perf_model.get_std(xy2perf[k]) for k in xy2perf}
    # xy2sorted: {(sigma1, simga2): [(candidate, perf), (candidate, perf), ...]}
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

    xy2normed_p, xy2normed_r = {}, {}
    for coord in xy2perf:
        logging.info('{}:\nBest Certain Design: {} ({}) ({}, {})\nPerf Opt Design: {} ({}) ({} <- {}, {})\nRisk Opt Design: {} ({}) ({}, {} <- {})'.format(
            coord,
            xy2best_design[(0, 0)], Metrics.h_metric(xy2best_design[(0, 0)]), 
            xy2mean[coord][xy2best_design[(0, 0)]], xy2risk[coord][xy2best_design[(0, 0)]], 
            xy2best_design[coord], Metrics.h_metric(xy2best_design[coord]),
            xy2mean[coord][xy2best_design[coord]], xy2mean[coord][xy2best_design[(0, 0)]], 
            xy2risk[coord][xy2best_design[coord]],
            xy2risk_best_design[coord], Metrics.h_metric(xy2risk_best_design[coord]),
            xy2mean[coord][xy2risk_best_design[coord]], xy2risk[coord][xy2risk_best_design[coord]],
            xy2risk[coord][xy2best_design[(0, 0)]]))

        conv_design = xy2best_design[(0, 0)]
        # Last element is area_left.
        conv_candidate = conv_design[:-1]
        if not reevaluate:
            conv_perf = xy2mean[coord][conv_design]
            conv_risk = xy2risk[coord][conv_design]
        else:
            d2perf = evaluate_with_uncertainty(perf_model, coord[0], coord[1], conv_candidate)
            conv_perf = perf_model.get_mean(d2perf)[conv_candidate]
            conv_risk = perf_model.get_risk(xy2best_mean[coord], d2perf)[conv_candidate]
        perf_opt_design = xy2best_design[coord]
        perf_opt_candidate = perf_opt_design[:-1]
        risk_opt_design = xy2risk_best_design[coord]
        risk_opt_candidate = risk_opt_design[:-1]
        if not reevaluate:
            perf_pdesign = xy2mean[coord][perf_opt_design]
            risk_pdesign = xy2risk[coord][perf_opt_design]
            perf_rdesign = xy2mean[coord][risk_opt_design]
            risk_rdesign = xy2risk[coord][risk_opt_design]
        else:
            d2perf = evaluate_with_uncertainty(perf_model, coord[0], coord[1], perf_opt_candidate)
            perf_pdesign = perf_model.get_mean(d2perf)[perf_opt_candidate]
            risk_pdesign = perf_model.get_risk(xy2best_mean[coord], d2perf)[perf_opt_candidate]
            d2perf = evaluate_with_uncertainty(perf_model, coord[0], coord[1], risk_opt_candidate)
            perf_rdesign = perf_model.get_mean(d2perf)[risk_opt_candidate]
            risk_rdesign = perf_model.get_risk(xy2best_mean[coord], d2perf)[risk_opt_candidate]
        xy2normed_p[coord] = (perf_pdesign/conv_perf, risk_pdesign/conv_risk if conv_risk else 1.)
        xy2normed_r[coord] = (perf_rdesign/conv_perf, risk_rdesign/conv_risk if conv_risk else 1.)
        logging.debug('Coord ({}, {}):\nconv_perf: {} ({})\nperf_pdesign: {} ({})'.format(
            coord[0], coord[1], conv_perf, conv_design, perf_pdesign, perf_opt_design))
        logging.debug('Coord ({}, {}):\nconv_risk: {} ({})\nrisk_rdesign: {} ({})'.format(
            coord[0], coord[1], conv_risk, conv_design, risk_rdesign, risk_opt_design))
    return xy2normed_p, xy2normed_r

def prepare_plot(perf_model, xy2z_gt, xy2z_ap):
    xy2normed_gt_p, xy2normed_gt_r = get_xy2normed_perf(perf_model, xy2z_gt)
    xy2normed_ap_p, xy2normed_ap_r = get_xy2normed_perf(perf_model, xy2z_ap, True)

    with open(args.pickle, 'w') as f:
        data = (xy2normed_gt_p, xy2normed_gt_r, xy2normed_ap_p, xy2normed_ap_r)
        pickle.dump(data, f)
        f.close()

    return xy2normed_gt_p, xy2normed_gt_r, xy2normed_ap_p, xy2normed_ap_r

def plot_optimal_surface(xy2normed_gt_p, xy2normed_gt_r, xy2normed_ap_p, xy2normed_ap_r):
    X, Y = [], []
    for coord in xy2normed_gt_p:
        if coord[0] not in X:
            X.append(coord[0])
        if coord[1] not in Y:
            Y.append(coord[1])
    X = sorted(X)
    Y = sorted(Y)
    Z_perf_gt_p, Z_perf_ap_p = [], []
    Z_perf_gt_r, Z_perf_ap_r = [], []
    for x in X:
        tmp_gt_p, tmp_ap_p = [], []
        tmp_gt_r, tmp_ap_r = [], []
        for y in Y:
            coord = (x, y)
            tmp_gt_p.append(xy2normed_gt_p[coord][0])
            tmp_ap_p.append(xy2normed_ap_p[coord][0])
            tmp_gt_r.append(xy2normed_gt_r[coord][1])
            tmp_ap_r.append(xy2normed_ap_r[coord][1])
            logging.debug('Coord ({}, {}):\nNormlized perf: {}, {}\nNormlized risk: {}, {}'.format(
                x, y, tmp_gt_p[-1], tmp_ap_p[-1], tmp_gt_r[-1], tmp_ap_r[-1]))
        Z_perf_gt_p.append(tmp_gt_p)
        Z_perf_ap_p.append(tmp_ap_p)
        Z_perf_gt_r.append(tmp_gt_r)
        Z_perf_ap_r.append(tmp_ap_r)

    if not args.fc_file and not args.cpudb_dir:
        title = 'fake_3d_' + str(args.f) + '_' + str(args.c)
    elif not args.fc_file:
        title = 'cpudb_3d_' + str(args.f) + '_' + str(args.c)
    elif not args.cpudb_dir:
        title = 'fake_3d_' + args.fc_file[args.fc_file.rfind('/')+1:args.fc_file.rfind('.')]
    else:
        title = 'cpudb_3d_' + args.fc_file[args.fc_file.rfind('/')+1:args.fc_file.rfind('.')]
    
    #PlotHelper.plot_3d_surface(X, Y, Z_perf_gt_p, title)
    PlotHelper.plot_3d_surfaces([X, X], [Y, Y], [Z_perf_gt_p, Z_perf_ap_p], ['gt', 'ap'], title)
    PlotHelper.plot_3d_surfaces([X, X], [Y, Y], [Z_perf_gt_r, Z_perf_ap_r], ['gt', 'ap'], 'risk' + title)

def load_from_xyz():
    with open(args.load_xyz, 'r') as f:
        data = pickle.load(f)
        f.close()
    return data

def load_from_files():
    assert args.load_path_gt and args.load_path_ap and args.load_path_gt != args.load_path_ap
    with open(args.load_path_gt, 'r') as f:
        logging.debug('Loading from gt {}'.format(args.load_path_gt))
        xy2z_gt = pickle.load(f)
        f.close()
    with open(args.load_path_ap, 'r') as f:
        logging.debug('Loading from ap {}'.format(args.load_path_ap))
        xy2z_ap = pickle.load(f)
        f.close()
    logging.debug('Loaded.')
    return xy2z_gt, xy2z_ap

def main():
    parse_args()
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)

    risk_func = RiskFunctionCollection.risk_function_collection[args.risk_func]
    perf_model = PerformanceModel(args.math_model, risk_func) 

    if args.load_xyz:
        data = load_from_xyz()
    else:
        assert args.load
        xy2z_gt, xy2z_ap = load_from_files()
        data = prepare_plot(perf_model, xy2z_gt, xy2z_ap)
    plot_optimal_surface(*data)
            
def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel',
            help='set log level.')
    parser.add_argument('--math_model', action='store', default='symmetric',
            help='which math model to use: symmetric, asymmetric or dynamic')
    parser.add_argument('--risk_func', action='store', dest='risk_func', default='linear',
            help='select risk model to use: step, linear or quad.')
    parser.add_argument('--f', action='store', type=float, default=.0,
            help='Fixed f value to use.')
    parser.add_argument('--c', action='store', type=float, default=.0,
            help='Fixed c value to use.')
    parser.add_argument('--fc-file', action='store', dest='fc_file', default=None,
            help='filepath to empirical workload for fc regression.')
    parser.add_argument('--cpudb-dir', action='store', dest='cpudb_dir', default=None,
            help='path to cpudb directory.')
    parser.add_argument('--trans', action='store_true', dest='trans', default=False,
            help='use transformed Gaussian as inputs.')
    parser.add_argument('--load', action='store_true', default=False,
            help='Use previouse results.')
    parser.add_argument('--pickle', action='store', dest='pickle', default='xyz.pickle',
            help='Path to store result.')
    parser.add_argument('--load-xyz', action='store', dest='load_xyz', default=None,
            help='Load from last result.')
    parser.add_argument('--load-path-gt', action='store', default=None,
            help='File path to load ground truth result.')
    parser.add_argument('--load-path-ap', action='store', default=None,
            help='File path to load approximation result.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
