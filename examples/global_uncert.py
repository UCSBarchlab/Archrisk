from base.sheet import *
from base.helpers import *
from models.abstract_application import App, AppHelper
from models.custom_distributions import CustomDistribution
from models.regression_models import HillMartyModel, ExtendedHillMartyModel
from models.performance_models import MathModel
from models.risk_functions import RiskFunctionCollection
from models.uncertainty_models import UncertaintyModel
from models.regression_models import ExtendedHillMartyModel
from utils.kde import KDE, Transformations
from utils.preprocessing import FileHelper
from utils.plotting import PlotHelper

from collections import defaultdict
import argparse
import functools
import logging
import math
import matplotlib
import itertools
import pickle

kZeroThreshold = 1e-2

class PRPoint(object):
    def __init__(self, design, perf, risk):
        self.design = design
        self.perf = perf
        self.risk = risk

class PerformanceModel(MathModel):
    area = 256
    perf_target = 'speedup'
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
        return {k:self.risk_func.get_risk(ref, v) for k, v in d2perf.iteritems()}

    def get_mean(self, d2uc):
        """ Extracts mean performance.
        """
        return {k:self.get_numerical(v) for k, v in d2uc.iteritems()}

    def get_uncert(self, d2uc):
        return {k:math.sqrt(self.get_var(v)) for k, v in d2uc.iteritems()}

    def apply_candidate(self, candidate, num_func, perf_func, app):
        """ Try on a certain given design candidate.

        Args:
            candidate: design point
            num_func: function to generate core number distributions.
            perf_func: function to generate core performance distributions.
            app: application to solve.

        Returns:
            perf: result performance distribution.
        """
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
        logging.debug('perf: {}'.format(perf))
        return perf

    def print_latex(self):
        self.sheet1.printLatex()

args = None

def plot(d2perf, perf_model):
    d2mean = perf_model.get_mean(d2perf)
    d2var = perf_model.get_var(d2perf)
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
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)
    parse_args()

    risk_func = RiskFunctionCollection.risk_function_collection[args.risk_func]
    perf_model = PerformanceModel(args.math_model, risk_func) 

    if args.load:
        load_from_file(perf_model)

    #===============
    # Example Usage
    #===============
    else:
        sigmas = np.arange(.0, 1.1, .1)
        # Design constant.
        design_uncert_scale = .1

        # Fabrication constant.
        fab_d = .003
        fab_a = .5

        if args.fc_file:
            apps = AppHelper.gen_app(args.fc_file, ExtendedHillMartyModel())
            fs = [app.f for app in apps]
            cs = [app.c for app in apps]
            f_pdf = KDE.fit(fs, Transformations.logit)
            c_pdf = KDE.fit(cs, np.log)
            f_dist = CustomDistribution.EmpiricalDistribution(f_pdf, Transformations.sigmoid)
            c_dist = CustomDistribution.EmpiricalDistribution(c_pdf, np.exp)
            logging.debug('Distribution of f: ({}, {})'.format(f_dist.mean, np.sqrt(f_dist.var)))
            logging.debug('Distribution of c: ({}, {})'.format(c_dist.mean, np.sqrt(c_dist.var)))
            # Generate 'average' app.
            app = App('Average', f_dist.mean, c_dist.mean)
        else:
            apps = AppHelper.gen_app()
            app = App('Fake', args.f, args.c)

        if not args.apply_design:
            raise ValueError('No design candidate provided.')

        design = tuple(args.apply_design)
        mean_f, mean_c = app.f, app.c
        sigma2perf = {}

        # Get const perf first.
        perf_func = lambda x: math.sqrt(x)
        num_func = lambda x, y: y/x
        const_perf = perf_model.apply_candidate(design, num_func, perf_func, app)
        sigma2perf[0.0] = const_perf

        # Get distributional perf.
        # Get singluar sigmas.
        logging.debug('====== f only ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            if sigma > .0:
                if not args.trans:
                    app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                        mean_f, sigma * (1 - mean_f)))
                else:
                    gt_f = CustomDistribution.NormalizedBinomialDistribution(
                            mean_f, sigma * (1 - mean_f))
                    app.set_f(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_f, sigma * (1 - mean_f), gt_f._mcpts, lower=0, upper=1))
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf
        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_f_only = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_f_only = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        logging.debug('====== c only ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            if sigma > .0:
                if not args.trans:
                    app.set_c(CustomDistribution.NormalizedBinomialDistribution(
                        mean_c, sigma * mean_c))
                else:
                    gt_c = CustomDistribution.NormalizedBinomialDistribution(
                            mean_c, sigma * mean_c)
                    app.set_c(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_c, sigma * mean_c, gt_c._mcpts, lower=0))
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf
        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_c_only = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_c_only = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        logging.debug('====== design only ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            perf_func = lambda x: math.sqrt(x)
            if sigma > .0:
                perf_func = UncertaintyModel.core_design_performance_uncertainty(
                        sigma * design_uncert_scale, .0000001)
            num_func = lambda x, y: y/x
            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf
        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_design_only = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_design_only = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        logging.debug('====== perf only ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            perf_func = lambda x: math.sqrt(x)
            if sigma > .0:
                if not args.trans:
                    perf_func = UncertaintyModel.core_performance_uncertainty(sigma)
                else:
                    perf_func = UncertaintyModel.core_performance_boxcox_uncertainty(sigma)
            num_func = lambda x, y: y/x
            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf
        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_perf_only = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_perf_only = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}
        
        logging.debug('====== fab only ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            if sigma > .0:
                num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)
            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf
        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_fab_only = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_fab_only = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        # Get full sigma.
        logging.debug('====== all ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            num_func = lambda x, y: y/x
            perf_func = lambda x: math.sqrt(x)
            if sigma > .0:
                if not args.trans:
                    app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                        mean_f, sigma * (1 - mean_f)))
                    app.set_c(CustomDistribution.NormalizedBinomialDistribution(
                        mean_c, sigma * mean_c))
                    perf_func = UncertaintyModel.core_design_performance_uncertainty(
                            sigma * design_uncert_scale, sigma)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)
                else:
                    gt_f = CustomDistribution.NormalizedBinomialDistribution(
                            mean_f, sigma * (1-mean_f))
                    app.set_f(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_f, sigma * (1-mean_f), gt_f._mcpts, lower=0, upper=1))
                    gt_c = CustomDistribution.NormalizedBinomialDistribution(
                            mean_c, sigma * mean_c)
                    app.set_c(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_c, sigma * mean_c, gt_c._mcpts, lower=0))
                    perf_func = UncertaintyModel.core_design_performance_boxcox_uncertainty(
                            sigma * design_uncert_scale, sigma)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)
            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf

        # hists for distributional perf.
        #sigma2perfs = {k: sigma2perf[k]._mcpts for k in np.arange(.1, 1., .1)}
        #PlotHelper.plot_hists(sigma2perfs)

        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_full = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_full = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}
        
        
        logging.debug('====== no f ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            if sigma > .0:
                if not args.trans:
                    #app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                    #mean_f, sigma/2 * (1 - mean_f)))
                    app.set_c(CustomDistribution.NormalizedBinomialDistribution(
                        mean_c, sigma * mean_c))
                    perf_func = UncertaintyModel.core_design_performance_uncertainty(
                            sigma * design_uncert_scale, sigma)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d) 
                else:
                    #gt_f = CustomDistribution.NormalizedBinomialDistribution(
                    #        mean_f, sigma * (1-mean_f))
                    #app.set_f(CustomDistribution.BoxCoxTransformedDistribution(
                    #    mean_f, sigma * (1-mean_f), gt_f._mcpts))
                    gt_c = CustomDistribution.NormalizedBinomialDistribution(
                            mean_c, sigma * mean_c)
                    app.set_c(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_c, sigma * mean_c, gt_c._mcpts))
                    perf_func = UncertaintyModel.core_design_performance_boxcox_uncertainty(
                            sigma * design_uncert_scale, sigma)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d) 

            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf

        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_f = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_f = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        logging.debug('====== no c ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            if sigma > .0:
                if not args.trans:
                    app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                        mean_f, sigma * (1 - mean_f)))
                    #app.set_c(CustomDistribution.NormalizedBinomialDistribution(
                    #mean_c, sigma/2 * mean_c))
                    perf_func = UncertaintyModel.core_design_performance_uncertainty(
                            sigma * design_uncert_scale, sigma)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d) 
                else:
                    gt_f = CustomDistribution.NormalizedBinomialDistribution(
                            mean_f, sigma * (1-mean_f))
                    app.set_f(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_f, sigma * (1-mean_f), gt_f._mcpts))
                    #gt_c = CustomDistribution.NormalizedBinomialDistribution(
                    #        mean_c, sigma * mean_c)
                    #app.set_c(CustomDistribution.BoxCoxTransformedDistribution(
                    #    mean_c, sigma * mean_c, gt_c._mcpts))
                    perf_func = UncertaintyModel.core_design_performance_boxcox_uncertainty(
                            sigma * design_uncert_scale, sigma)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)

            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf

        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_c = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_c = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        logging.debug('====== no design ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            if sigma > .0:
                if not args.trans:
                    app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                        mean_f, sigma * (1 - mean_f)))
                    app.set_c(CustomDistribution.NormalizedBinomialDistribution(
                        mean_c, sigma * mean_c))
                    #perf_func = UncertaintyModel.core_design_performance_uncertainty(
                    #        sigma * design_uncert_scale, sigma)
                    perf_func = UncertaintyModel.core_performance_uncertainty(sigma)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)
                else:
                    gt_f = CustomDistribution.NormalizedBinomialDistribution(
                            mean_f, sigma * (1-mean_f))
                    app.set_f(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_f, sigma * (1-mean_f), gt_f._mcpts))
                    gt_c = CustomDistribution.NormalizedBinomialDistribution(
                            mean_c, sigma * mean_c)
                    app.set_c(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_c, sigma * mean_c, gt_c._mcpts))
                    #perf_func = UncertaintyModel.core_design_performance_boxcox_uncertainty(
                    #        sigma * design_uncert_scale, sigma)
                    perf_func = UncertaintyModel.core_performance_boxcox_uncertainty(sigma)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)

            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf

        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_design = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_design = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        logging.debug('====== no perf ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            if sigma > .0:
                if not args.trans:
                    app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                        mean_f, sigma * (1 - mean_f)))
                    app.set_c(CustomDistribution.NormalizedBinomialDistribution(
                        mean_c, sigma * mean_c))
                    #perf_func = UncertaintyModel.core_design_performance_uncertainty(
                    #        sigma * design_uncert_sacle, sigma)
                    perf_func = UncertaintyModel.core_design_performance_uncertainty(
                            sigma * design_uncert_scale, .0000001)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d) 
                else:
                    gt_f = CustomDistribution.NormalizedBinomialDistribution(
                            mean_f, sigma * (1-mean_f))
                    app.set_f(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_f, sigma * (1-mean_f), gt_f._mcpts))
                    gt_c = CustomDistribution.NormalizedBinomialDistribution(
                            mean_c, sigma * mean_c)
                    app.set_c(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_c, sigma * mean_c, gt_c._mcpts))
                    #perf_func = UncertaintyModel.core_design_performance_boxcox_uncertainty(
                    #        sigma * design_uncert_scale, sigma)
                    perf_func = UncertaintyModel.core_design_performance_boxcox_uncertainty(
                            sigma * design_uncert_scale, .0000001)
                    num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)

            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf

        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_perf = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_perf = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        logging.debug('====== no fab ======')
        for sigma in sigmas:
            logging.debug('Sigma: {}'.format(sigma))
            app = App('fake', mean_f, mean_c)
            perf_func = lambda x: math.sqrt(x)
            num_func = lambda x, y: y/x
            if sigma > .0:
                if not args.trans:
                    app.set_f(CustomDistribution.NormalizedBinomialDistribution(
                        mean_f, sigma * (1 - mean_f)))
                    app.set_c(CustomDistribution.NormalizedBinomialDistribution(mean_c, sigma * mean_c))
                    perf_func = UncertaintyModel.core_design_performance_uncertainty(
                            sigma * design_uncert_scale, sigma)
                    #num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)
                else:
                    gt_f = CustomDistribution.NormalizedBinomialDistribution(
                            mean_f, sigma * (1-mean_f))
                    app.set_f(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_f, sigma * (1-mean_f), gt_f._mcpts))
                    gt_c = CustomDistribution.NormalizedBinomialDistribution(
                            mean_c, sigma * mean_c)
                    app.set_c(CustomDistribution.BoxCoxTransformedDistribution(
                        mean_c, sigma * mean_c, gt_c._mcpts))
                    perf_func = UncertaintyModel.core_design_performance_boxcox_uncertainty(
                            sigma * design_uncert_scale, sigma)
                    #num_func = UncertaintyModel.fabrication_uncertainty(fab_a, fab_d)

            perf = perf_model.apply_candidate(design, num_func, perf_func, app)
            sigma2perf[sigma] = perf

        sigma2mean = perf_model.get_mean(sigma2perf)
        sigma2var = perf_model.get_uncert(sigma2perf)
        #sigma2normed_var_fab = {s: sigma2var[s]/const_perf for s in sigmas}
        sigma2normed_var_fab = {s: (sigma2mean[s], sigma2var[s]) for s in sigmas}

        legends1 = ['no f', 'no c', 'no design', 'no perf', 'no fab', 'all']
        list1 = [sigma2normed_var_f, sigma2normed_var_c,
                sigma2normed_var_design, sigma2normed_var_perf,
                sigma2normed_var_fab, sigma2normed_var_full]
        title1 = 'oneout_uncert' if not args.trans else 'oneout_uncert_trans'
        PlotHelper.plot_trends(legends1, list1, const_perf, False, title1)
        PlotHelper.plot_trends(legends1, list1, const_perf, True, title1)
        

        legends2 = ['f only', 'c only', 'design only', 'perf only', 'fab only', 'all']
        list2 = [sigma2normed_var_f_only, sigma2normed_var_c_only,
                sigma2normed_var_design_only, sigma2normed_var_perf_only,
                sigma2normed_var_fab_only, sigma2normed_var_full]
        title2 = 'singlar_uncert' if not args.trans else 'singlar_uncert_trans'
        title2 = 'f_' + str(args.f) + '_c_' + str(args.c) + '_' + title2
        #PlotHelper.plot_trends(legends2, list2, const_perf, False, title2)
        #PlotHelper.plot_trends(legends2, list2, const_perf, True, title2)

        #PlotHelper.plot_trends(legends1 + legends2, list1 + list2, const_perf)

         
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
            help='File for empirical fc estimation.')
    parser.add_argument('--trans', action='store_true', default=False,
            help='Use BoxCox transformation.')
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
