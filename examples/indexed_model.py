from models.abstract_application import App, AppHelper
from models.performance_models import PerformanceModel
from models.distributions import Distribution
import argparse
import functools
import logging
import numpy as np

def main():
    parse_args()
    app = App('fake', args.f, args.c)
    perf_model = PerformanceModel(args.math_model, args.risk_func)
    perf_model.add_index_bounds('i', upper=len(perf_model.designs))
    for i, d in enumerate(perf_model.designs):
        perf_model.add_given('core_perf_'+str(i), np.sqrt(d))
        perf_model.add_given('core_num_' + str(i), Distribution.LogNormalDistribution(1, .1))
    logging.debug('Eval result: {}'.format(perf_model.compute(app)['speedup']))

def parse_args():
    global args

    np.set_printoptions(precision=4, threshold='nan')
    parser = argparse.ArgumentParser()

    parser.add_argument('--log', action='store', dest='loglevel',
            help='set log level.')
    parser.add_argument('--math-model', action='store', default='symmetric',
            help='which math model to use: symmetric, asymmetric or dynamic')
    parser.add_argument('--risk-func', action='store', dest='risk_func', default='linear',
            help='select risk model to use: step, linear or quad.')
    parser.add_argument('--f', action='store', type=float, default=.0,
            help='Fixed f value to use.')
    parser.add_argument('--c', action='store', type=float, default=.0,
            help='Fixed c value to use.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
