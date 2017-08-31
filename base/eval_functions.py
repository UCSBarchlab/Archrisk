from sympy import *
from mcerp import *
import mcerp as mcp
import numpy as np
import logging

from models.distributions import Distribution
from models.uncertainty_models import UncertaintyModel

def FABRIC(size, design_num):
    """ Compute the fabricated number of cores.
    """
    logging.debug('FABRIC -- size: {}, design_num: {}'.format(size, design_num))
    assert size >= 0 and design_num >= 0, 'FABRIC -- size: {}, num: {}'.format(size, design_num)
    num_func = UncertaintyModel.fabrication_boxcox()
    return num_func(size, design_num)

def PERF(arg):
    """ Compute the performance of core.
    """
    assert arg >= 0, 'PERF -- Core size {} is negative'.format(arg)
    perf_func = UncertaintyModel.core_perf_boxcox()
    return perf_func(arg)

def SUM(*arg):
    """ Take the sum of the input list.
    """
    arg = np.asarray(arg) * np.ones(1)
    return reduce(lambda x,y: x+y, arg)

def __has_distribution(args):
    """ Return true if args has at least one distribution-typed element.
    """

    ts = [isinstance(arg, UncertainFunction) for arg in args]
    return any(ts)

def CONDMAX(*arg):
    """ Conditional maximum function.
        Conputes a rv X, consisting of the maximum of performance
        if corresponding num is greater than 0 for each MC trial.

        Args:
            arg: Sympy parsed arguments, [n0, p0, n1, p1, ...]
            each n, p can be either a distribution or a scalar.
        Return:
            x: result distribution.
    """

    ns = arg[::2]
    ps = arg[1::2]
    assert ns and ps and len(ns) == len(ps)

    if not __has_distribution(ns) and not __has_distribution(ps):
        candidate = [p for n, p in zip(ns, ps) if n > 0]
        return max(candidate)
    else:
        x = Distribution.GetDummy()
        mcpts = [None] * mcp.npts
        for i in range(mcp.npts):
            candidate = [p._mcpts[i] if isinstance(p, UncertainFunction) else p
                    for n, p in zip(ns, ps)
                    if (n._mcpts[i] if isinstance(n, UncertainFunction) else n) > 0]
            mcpts[i] = max(candidate) if candidate else Distribution.NON_ZERO_FACTOR
        x._mcpts = np.asarray(mcpts)
        return x
