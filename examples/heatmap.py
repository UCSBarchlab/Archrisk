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

args = None

def get_xy2normed_perf(perf_model, xy2z, reevaluate=False):
    # xy2z is actually (x, y)(design) -> perf, so xy2z[x, y][d] -> perf.
    if reevaluate:
        logging.info('===Reevaluating===')
    assert (0, 0) in xy2z
    xy2perf = xy2z
    xy2mean = {k: perf_model.get_mean(xy2perf[k]) for k in xy2perf}
    xy2std = {k: perf_model.get_std(xy2perf[k]) for k in xy2perf}
    # xy2sorted: {(sigma1, simga2): [(candidate, perf), (candidate, perf), ...]}
    xy2sorted = {coord: sorted(xy2mean[coord].items(), key=lambda (k, v): v, reverse=True) for coord in xy2mean}
    xy2best_mean = {coord: xy2sorted[coord][0][1] for coord in xy2sorted}
    xy2risk = {k: perf_model.get_risk(xy2best_mean[k], xy2perf[k]) for k in xy2perf}
    xy2risk_sorted = {coord: sorted(xy2risk[coord].items(), key=lambda (k, v): v) for coord in xy2risk}
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
    with open('xyz.pickle', 'r') as f:
        data = pickle.load(f)
        f.close()
    return data

def load_from_files():
    assert args.load_path_gt and args.load_path_ap and args.load_path_gt != args.load_path_ap
    with open(args.load_path_gt, 'r') as f:
        logging.debug('Loading from gt {}'.format(args.load_path_gt))
        xy2z_gt = pickle.load(f)
        f.close()
    logging.debug('Loaded.')
    return xy2z_gt

def main():
    parse_args()
    matplotlib.rc('xtick', labelsize=18)
    matplotlib.rc('ytick', labelsize=18)

    assert args.load
    xy2z_gt = load_from_files()
    data = prepare_plot(xy2z_gt)
    plot_heatmap(*data)
            
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
    parser.add_argument('--load-xyz', action='store_true', dest='load_xyz', default=False,
            help='Load from last result.')
    parser.add_argument('--load-path-gt', action='store', default=None,
            help='File path to load ground truth result.')

    args = parser.parse_args()
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: {}'.format(args.loglevel))
    logging.basicConfig(format='%(levelname)s: %(message)s', level=numeric_level)

if __name__ == '__main__':
    main()
