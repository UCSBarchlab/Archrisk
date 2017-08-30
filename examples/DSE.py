from base.sheet import *
from base.helpers import *
from collections import defaultdict
from models.abstract_application import App
from models.distributions import Distribution
from models.math_models import MathModel
from models.performance_models import PerformanceModel
from models.risk_functions import RiskFunctionCollection
from models.uncertainty_models import UncertaintyModel
import argparse
import functools
import math
import matplotlib
import logging
import itertools
import pickle

args = None

def main():
    parse_args()
    
    # Create risk function.
    risk_func = RiskFunctionCollection.risk_function_collection[args.risk_func]

    # Create executable high-level performance model.
    perf_model = PerformanceModel(args.math_model, risk_func) 
    
    # Application constants.
    mean_f, mean_c = args.f, args.c

    # Design uncertainty constants.
    design_risk_scale = .1

    # Set application type.
    app = App('fake', mean_f, mean_c)

    # To store results.
    xy2z = defaultdict()

    # Loop through input uncertainty levels.
    sigmas = np.arange(0, 1., .2)
    for sigma1 in sigmas:
        for sigma2 in sigmas:
            logging.info('(sigma_app, sigma_arch): ({}, {})'.format(sigma1, sigma2))

            # Setting the architecture uncertainty level.
            UncertaintyModel.set_rates(sigma2*design_risk_scale, sigma2)

            # Setting the application uncertainty level.
            app = App('fake', mean_f, mean_c)
            if not args.trans:
                # Use groundtruth distributions.
                if sigma1 > .0:
                    app.set_f(Distribution.NormalizedBinomialDistribution(
                        mean_f, sigma1 * (1-mean_f)))
                    app.set_c(Distribution.NormalizedBinomialDistribution(
                        mean_c, sigma1 * mean_c))
            else:
                # Use transformed distributions.
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
            # Actual DSE.
            d2uncert_perf = perf_model.get_perf(app)
            xy2z[(sigma1, sigma2)] = d2uncert_perf
            
def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel',
            help='Set log level.')
    parser.add_argument('--math-model', action='store', default='symmetric',
            choices=MathModel.names(),
            help='Select math model to use.')
    parser.add_argument('--risk-func', action='store', dest='risk_func',
            default='linear', choices=RiskFunctionCollection.risk_function_collection,
            help='Select risk function to use.')
    parser.add_argument('--f', action='store', type=float, default=.9,
            help='Set f value to use.')
    parser.add_argument('--c', action='store', type=float, default=.01,
            help='Set c value to use.')
    parser.add_argument('--trans', action='store_true', dest='trans', default=False,
            help='Use transformed distributions.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
