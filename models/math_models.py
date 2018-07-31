from base.sheet import *
from base.helpers import *
from base.eval_functions import * 
from mcerp import *
from uncertainties.core import AffineScalarFunc
import logging
import numpy as np
import re

class MathModel(object):
    """ Defines the mathematical symbols and expressions used for modelling.
    """
    
    #==========================================================================
    # define mapping between custom Symbols -> eval functions
    #==========================================================================
    custom_funcs = {'FABRIC': FABRIC, 'PERF': PERF,
            'SUM': SUM, 'CONDMAX': CONDMAX}

    #==========================================================================
    # define all symbols in the system
    #==========================================================================
    index_syms = ['i']

    config_syms = ['core_design_size_i', 'core_design_num_i',
            'core_size_i', 'core_num_i', 'core_perf_i']

    perf_syms = ['f', 'c', 't_s', 't_p']

    stat_syms = ['execute_time', 'speedup']

    #============================================================================
    # define system equations
    #============================================================================
     
    common_exprs = [ 
            'execute_time = t_s + t_p',
            'speedup = 1/execute_time',
            ]
     
    hete_exprs = [
            'core_size_i = core_design_size_i',
            'core_num_i = FABRIC(core_size_i, core_design_num_i)',
            'core_perf_i = PERF(core_size_i)',
            't_s = (1-f+SUM([core_num_i])*c)/CONDMAX({core_num_i, core_perf_i})',
            't_p = f/SUM([core_num_i * core_perf_i])',
            ]

    symm_exprs = [
            'core_perf = sqrt(core_size)',
            't_s = (1 - f + core_num * c)/core_perf',
            't_p = f/(core_num * core_perf)',
            'core_num = area_total / core_size',
            ]

    asymm_exprs = [
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f +  (1 + base_core_num) * c)/small_core_perf',
            't_p = f / (base_core_num + small_core_perf)',
            'base_core_num = area_total - small_core_size'
            ]

    dynamic_exprs = [
            'small_core_perf = sqrt(small_core_size)',
            't_s = (1 - f + area_total * c)/small_core_perf',
            't_p = f/area_total'
            ]

    def get_numerical(self, result):
        if isinstance(result, UncertainFunction):
            return result.mean
        elif isinstance(result, AffineScalarFunc):
            return result.n
        else:
            return result

    def get_var(self, result):
        if isinstance(result, UncertainFunction):
            return result.var
        elif isinstance(result, AffineScalarFunc):
            return result.std_dev * result.std_dev
        else:
            logging.warn('Trying to get var on constant: {}'.format(result))
            return 0

    @staticmethod
    def names():
        return ['symmetric', 'asymmetric', 'hete', 'dynamic']
