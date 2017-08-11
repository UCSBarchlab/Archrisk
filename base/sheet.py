""" This provides all low-level functionalities.
"""

from copy import copy, deepcopy
from functools import partial
from helpers import *
from mcerp import *
from parser import Parser
from sympy import *
from sympy.printing.latex import print_latex
from sympy.utilities.lambdify import lambdify, lambdastr
from scipy.optimize import minimize
from uncertainties import umath as a_umath
from uncertainties import wrap as uwrap
from uncertainties import ufloat
from utils.gaussian_decomposition import gaussian_decomposition
from utils.softmax import SoftMaximum
import logging
import numpy as np
import re

class PredType(object):
    """ Defines types of predicates.

        A predicate can be one of the following three:
        1). A given variable with fixed value, i.e. inputs to the model.
        2). An intermediate variable which is generated during evaluation using
            relies on enternal distributional information, i.e. uncertain variables.
        3). A response variable computed during evaluation, i.e. outputs of the model.
    """

    GIVEN = 1
    INTERMEDIATE = 2
    RESPONSE = 3

class Sheet(object):
    """ Sheet class automates expression manipulatin/expansion,
        error propagation and uncertainty injection.
    """

    def __init__(self, analytical=False, tag=None):
        # Custom function mappings used for lamdify.
        # TODO: this should be handled by addFuncs now.
        if analytical:
            self.sym2func = {"ceiling":a_umath.ceil, "Max":uwrap(SoftMaximum)}
            self.conv2analytical = self.conv2analytical_simple_compression
            #self.conv2analytical = self.conv2analytical_GMM
        else:
            self.sym2func = {"ceiling":umath.ceil}
        self.tag = tag
        self.analytical = analytical # Use analytical methods or MC.
        self.idx_bounds = {} # Bounds for index symbols, key type: string
        self.syms = {} # Symbolics used in system modeling, key type: string
        self.exprs_str = [] # Original string representation of expressions.
        self.exprs = [] # Sympy understandable parsed expressions.
        self.given = {} # Inputs to model evaluation, key type: symbol
        self.intermediates = {} # Uncertain dependent variables, key type: symbol
        self.response = set() # set of symbols.
        self.ordered_given = [] # List of symbols.
        self.sol_inter_set = {} # Intermediate solution set, key type: symbol
        self.inter_funcs = {} # Lamdified callables for intermediate vars, key type: symbol
        self.sol_final_set = {} # Final solution set, key type: symbol
        self.target_funcs = {} # Lamdified callables for response vars, key type: symbol
        self.opts = [] # varibles left to optimize.
        self.parser = Parser()
        npts = 100000

    def dump(self):
        print self.exprs

    def addSyms(self, sym_list):
        """ Adds symbols.

        Args:
            sym_list: [string].
        """
        self.syms.update(SympyHelper.initSyms(sym_list))

    def addFuncs(self, func_dict):
        """ Adds custom functions.
        """

        self.syms.update(SympyHelper.initFuncs(func_dict.keys()))
        self.sym2func.update(func_dict)

    def addExprs(self, expr_list):
        """Add equations in system.

        Args:
            expr_list: [string], all symbols mush have been defined with addSyms.
        """
        #self.exprs += SympyHelper.initExprs(expr_list, self.syms)
        self.exprs_str = expr_list

    def _predSanityCheck(self, t, predType):
        if predType is PredType.GIVEN:
            assert len(t) == 2 
            assert(isinstance(t[0], str) and (isinstance(t[1], float)
                or isinstance(t[1], UncertainFunction)))
        elif predType is PredType.INTERMEDIATE:
            assert len(t) == 2 
            assert(isinstance(t[0], str) and isinstance(t[1], partial))
        elif predType is PredType.RESPONSE:
            assert isinstance(t, str)
            t = [t]
        else:
            raise ValueError("pred type of %r not defined!" % t[0])
        if not t[0] in self.syms.keys():
            raise ValueError("%r not defined!" % t[0])

    # new values will overwirte old ones
    def addPreds(self, given=None, bounds=None, intermediates=None, response=None):
        """ Adds predicates.

            Args:
                given: [(var, value)]
                bounds: {k: (lower, upper)} 
                intermediates: [(var, callable)], callable must have one single argument: mean
                response: [string], var to solve for.
        """
        if bounds:
            self.idx_bounds = dict([(k, bounds[k]) for k in bounds])
            self.syms.update(self.parser.expand_syms(self.idx_bounds, self.syms))
       
        if given:
            for t in given:
                self.given[self.syms[t[0]]] = t[1]
        
        if intermediates:
            for t in intermediates:
                self._predSanityCheck(t, PredType.INTERMEDIATE)
                self.intermediates[self.syms[t[0]]] = t[1]

        if response:
            for t in response:
                self._predSanityCheck(t, PredType.RESPONSE)
                self.response.add(self.syms[t])

    def conv2analytical_GMM(self, given):
        """ Converts given MC to a vector of Gaussians using GMM EM fitting.
        The conversion result of this function are a vector of KNOWN gaussians,
        so the collapsing with uncertainties package won't lose shape of the
        distribution at this point.
        """

        result = []
        for q in given:
            if isinstance(q, UncertainFunction):
                components = gaussian_decomposition(q)
                mix = 0
                for (pi, mu, sigma) in components:
                    mix += pi * ufloat(mu, sigma)
                logging.debug('Original Dist: {}, {}\nDecomposed Mix Dist: {}, {}'.format(
                    q.mean, (q.var)**.5, mix.n, mix.std_dev))
                result.append(mix)
            else:
                result.append(q)
        return result

    def conv2analytical_simple_compression(self, given):
        """
        Convertes given MC to analytical form compatible with uncertainties.
        """

        result = []
        for q in given:
            if isinstance(q, UncertainFunction):
                nominal = q.mean
                std = np.sqrt(q.var)
                result.append(ufloat(nominal, std))
            else:
                result.append(q)
        return result

    def optimize(self, ordered_given, q_ordered_given, maximize=False):
        """ Minimization on responses.

            Args:
                ordered_given: [var], free varibles in an ordered way,
                                  "constants" should be behind all optimization targets
                q_ordered_given: [float], values for "constant" free variables
                                      in the same ordred way as above
            Returns:
                opt_val: {var, opt_val}, dict holding opt_val for each optimizing var
        """

        sol_final_set = solve(self.exprs, exclude=ordered_given, check=False, manual=True)[0]

        init_guesses = []
        opt_val = {}
        for k in self.opts:
            init_guesses.append(4)
            opt_val[k] = 4

        target_funcs = {}
        for var in self.response:
            if maximize:
                target_funcs[var] = lambdify(tuple(ordered_given), -1 * sol_final_set[var])
            else:
                target_funcs[var] = lambdify(tuple(ordered_given), sol_final_set[var])
            # TODO: parameterize bounds
            result = minimize(target_funcs[var], init_guesses,
                    args=tuple(q_ordered_given), bounds=[(0.1, 16.1)])
            if not result.success:
                print result.message
            else:
                for (k, v) in zip(self.opts, result.x):
                    opt_val[k] = v

        logging.debug("Sheet -- minimization: {}".format(opt_val))
        return opt_val

    def compute(self, maximize=False, constraints=None):
        """ Solves the system and computes the responses.
        """

        # Expand expressions on first time.
        if not self.exprs:
            self.exprs = self.parser.expand(self.exprs_str, self.idx_bounds, self.syms)

        u_math = umath if not self.analytical else a_umath

        # Generate an ordering for inputs.
        if not self.ordered_given:
            for (k, v) in self.given.iteritems():
                self.ordered_given.append(k)
        q_ordered_given = []

        # Ordered given list fed to optimization, might be different from ordered_given.
        opt_ordered_given = []
        opt_q_ordered_given = []

        self.opts = []
        for (k, v) in self.given.iteritems():
            if isinstance(v, str) and v == 'opt':
                self.opts.append(k)
            else:
                opt_ordered_given.append(k)
                opt_q_ordered_given.append(v)

        # Do minimization if needed.
        if self.opts:
            opt_given = []
            for k in self.opts:
                opt_given.append(k)
            opt_ordered_given = opt_given + opt_ordered_given
            opt_val = self.optimize(opt_ordered_given, opt_q_ordered_given, maximize)

        # Assemble q_ordered_given according to ordered_given.
        for k in self.ordered_given:
            if isinstance(self.given[k], str) and self.given[k] == 'opt':
                q_ordered_given.append(opt_val[k])
            else:
                q_ordered_given.append(self.given[k])
   
        # Solve for intermediate solution set, use cached version if possible.
        if not self.sol_inter_set:
            self.sol_inter_set = solve(self.exprs, exclude=self.ordered_given,
                    check=False, manual=True)[0]

            """ Uncomment following code to print intermediate solution set.
            logging.debug('Sheet -- Partial Solutions:')
            for k, s in self.sol_inter_set.iteritems():
                logging.debug('Inter -- {}: {}'.format(k, s))
            """

        # Generate intermediate funcs, use cached version if possible.
        if not self.inter_funcs:
            for var in self.intermediates.keys():
                if var in self.sol_inter_set:
                    logging.debug('{} Lambdification -- {}'.format(var,
                        self.sol_inter_set[var]))
                    self.inter_funcs[var] = lambdify(tuple(self.ordered_given),
                            self.sol_inter_set[var],
                            modules=[self.sym2func, u_math])
                else:
                    print "WARN: ignoring %r, no solution found!" % var
        else:
            # Make sure that all intermediates have a solution.
            for k in self.intermediates.keys():
                assert(k in self.inter_funcs)

        # Before generating quantities for intermediates, 
        # if we are performing analytical analysis, 
        # we need to convert any MC given to analytical form when needed.
        if self.analytical:
            q_ordered_given = self.conv2analytical(q_ordered_given)
            for q in q_ordered_given:
                assert(not isinstance(q, UncertainFunction))

        # Organize intermediates based on an ordering.
        ordered_intermediates = []
        q_ordered_intermediates = []
        for (k, v) in self.intermediates.iteritems():
            if k in self.inter_funcs:
                ordered_intermediates.append(k)
                mean_intermediate = float(self.inter_funcs[k](*tuple(q_ordered_given)))
                assert(callable(v))
                q_ordered_intermediates.append(v(mean_intermediate))
                logging.debug('Sheet -- Inter {}: {}'.format(
                    ordered_intermediates[-1], q_ordered_intermediates[-1]))

        # Solve for final solution set, use cached version if possible.
        if not self.sol_final_set:
            self.sol_final_set = solve(self.exprs,
                    exclude=self.ordered_given + ordered_intermediates, check=False, manual=True)[0]

            """ Uncomment the following to print final solution set.
            logging.debug('Sheet -- Final Solutions:')
            for k, s in self.sol_final_set.iteritems():
                logging.debug('Final -- {}: {}'.format(k, s))
            """

        # Generate target funcs, use cached version if possible.
        if not self.target_funcs:
            for var in self.response:
                logging.debug('{} Lambdification -- {}'.format(var, 
                    lambdastr(tuple(self.ordered_given + ordered_intermediates),
                        self.sol_final_set[var])))
                self.target_funcs[var] = lambdify(tuple(self.ordered_given + ordered_intermediates),
                        self.sol_final_set[var], modules=[self.sym2func, u_math])
  
        # Before generating quantities for final responses,
        # we need to convert any MC intermediate quantities to analytical form when needed.
        if self.analytical:
            q_ordered_intermediates = self.conv2analytical(q_ordered_intermediates)
            for q in q_ordered_intermediates:
                assert(not isinstance(q, UncertainFunction))

        # Compute response.
        q_response = {}
        for var in self.response:
            logging.debug('Solving {}'.format(str(var)))
            perf = self.target_funcs[var](*tuple(q_ordered_given + q_ordered_intermediates))
            q_response[str(var)] = perf

        return q_response
    
    def printLatex(self):
        symbol_names = {}
        for var in self.given:
            symbol_names[var] = str(var)
        if self.intermediates:
            for var in self.intermediates:
                symbol_names[var] = str(var)
        for expr in self.exprs:
            print latex(expr, symbol_names=symbol_names)
        for var in self.response:
            print "{} = {}".format(str(var), latex(self.sol_final_set[var],
                symbol_names=symbol_names))
            print "{} = {}".format(str(var), latex(self.sol_final_set[var]))
