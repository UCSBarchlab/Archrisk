from base.sheet import *
from collections import defaultdict
from models.distributions import Distribution
from models.math_models import MathModel
from models.risk_functions import RiskFunctionCollection

import copy

class PerformanceModel(MathModel):
    """ Provides high-level APIs for experiments.
    """
    
    # We are fixing the total number of resources on chip.
    area = 256
    perf_target = 'speedup'
    energy_target = 'energy'

    # Here's our cadidate core designs.
    designs = [8, 16, 32, 64, 128, 256]

    def __init__(self, selected_model, risk_function,
            use_energy=False, analytical=False):
        self.sheet1 = Sheet(analytical)
        self.sheet2 = Sheet(analytical)
        all_syms = (MathModel.index_syms +
                MathModel.config_syms +
                MathModel.perf_syms +
                MathModel.stat_syms +
                MathModel.power_syms)
        self.sheet1.addSyms(all_syms)
        self.sheet2.addSyms(all_syms)
        self.sheet1.addFuncs(MathModel.custom_funcs)
        self.sheet2.addFuncs(MathModel.custom_funcs)
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

        self.sheet1.addExprs(MathModel.common_exprs + model)
        self.sheet2.addExprs(MathModel.common_exprs + model)

        self.given = defaultdict()
        self.index_bounds = defaultdict()
        self.ims = defaultdict()
        self.target = []
        self.risk_func = risk_function
        self.use_energy = use_energy

    def gen_feed(self, k2v):
        """ Generates feed string to calculation sheets from key-value pairs.
        """

        feed = []
        for k, v in k2v.iteritems():
            feed += [(k, v)]
        return feed

    def compute(self, sheet, app):
        logging.debug('Evaluating app: {}'.format(app.get_printable()))
        given = self.gen_feed(self.given) + app.gen_feed()
        ims = self.gen_feed(self.ims)
        logging.debug('PerfModel -- Given: {} -- Intermediates: {}'.format(
            given, {k: v.func.__name__ for k, v in self.ims.iteritems()}))
        sheet.addPreds(given=given, bounds=self.index_bounds,
                intermediates=ims, response=self.target)
        result = sheet.compute() # Map from response var -> value.
        return result

    def dump(self):
        self.sheet1.dump()
        print self.risk_func

    def add_index_bounds(self, idx_base, lower=0, upper=None):
        assert upper # can't be infinite
        self.index_bounds[idx_base] = (lower, upper)

    def add_given(self, name, val):
        self.given[name] = val

    def add_target(self, name):
        self.target.append(name)

    def clear_targets(self):
        self.target = []

    def add_uncertain(self, name, partial_func):
        self.ims[name] = partial_func

    def get_risk(self, ref, d2perf):
        """ Computes risk for d2perf w.r.t. ref

            Args:
                ref: reference performance bar
                d2perf: performance array

            Returns:
                map of design -> mean risk
        """
        return {k: self.risk_func.get_risk(ref, v) for k, v in d2perf.iteritems()}

    def get_mean(self, d2uc):
        """ Extracts mean performance.
        """
        return {k: self.get_numerical(v) for k, v in d2uc.iteritems()}

    def get_std(self, d2uc):
        """ Extracts performance std.
        """
        return {k: np.sqrt(self.get_var(v)) for k, v in d2uc.iteritems()}

    def apply_design(self, candidate, app):
        """ Evaluates a given design candidate.

            Args:
                candidate: design point
                app: application to solve.

            Returns:
                perf: result performance distribution.
        """
        assert len(candidate) == len(self.designs)
        area_left = self.__class__.area - sum([x * y for x, y in zip(candidate, self.designs)])
        assert area_left >= 0 # all designs are bounded by total area
        logging.info('PerfModel -- Evaluating design: {} ({})'.format(candidate, area_left))
        for i, size_i in enumerate(self.designs):
            self.add_given('core_design_size_'+str(i), size_i)
        # treat left over as an additional core
        self.add_given('core_design_size_'+str(len(self.designs)), area_left)
        for i, num_i in enumerate(candidate):
            self.add_given('core_design_num_'+str(i), num_i)
        # we either have exactly one addtional core or not
        self.add_given('core_design_num_'+str(len(self.designs)), 1 if area_left else 0)
        result = self.compute(self.sheet1, app)
        perf = result[self.perf_target]
        energy = result[self.energy_target] if self.use_energy else ''
        logging.info('PerfModel -- Result: {}, {}'.format(perf, energy))
        return perf

    def iter_through_design(self, d2perf, ith, stop, candidate, app):
        """ Places cores on chip, evaluates a chip when a candidate core placement has been selected.

            Args:
                d2perf: result, map of design -> perf distribution, updated in-place.
                ith: the i-th design under inspection.
                stop: when to stop placing more cores.
                candidate: current candidate core placement list.
                app: application to solve.
        """

        area_left = self.__class__.area - sum(
                [x * y for x, y in zip(candidate, self.designs)])
        assert(area_left >= 0) # all designs are bounded by total area
        if ith == stop:
            # found a candidate design
            tag = tuple(candidate + [area_left])
            d2perf[tag] = self.apply_design(candidate, app)
        else:
            # still trying to place cores
            d_cur = self.designs[ith]
            i = 0
            while (i < area_left/self.designs[ith]+1):
                candidate[ith] = i
                self.iter_through_design(d2perf, ith+1, stop, candidate, app)
                candidate[ith] = 0
                # double the number of cores every time.
                i = 1 if i == 0 else i << 1

    def compute_core_perfs(self, app):
        """ Calculates and caches performace of each type of core.
        """
        logging.info('Computing core performances...')
        self.clear_targets()
        for i, d in enumerate(self.designs):
            self.add_given('core_design_size_'+str(i), d)
            self.add_target('core_perf_'+str(i))
            if self.use_energy:
                self.add_target('core_power_'+str(i))
        result = self.compute(self.sheet2, app)
        for i, d in enumerate(self.designs):
            name = 'core_perf_'+str(i)
            self.add_given(name, result[name])
            logging.debug('Core perf: {}'.format(result[name]))
            if self.use_energy:
                name = 'core_power_'+str(i)
                self.add_given(name, result[name])
                logging.debug('Core power: {}'.format(result[name]))
        self.clear_targets()

    def get_perf(self, app):
        """ Computes peformance distribution over a single app.
        """
        d2perf = defaultdict()
        n_designs = len(self.designs)
        candidate = [0] * n_designs
        self.add_index_bounds('i', upper=len(self.designs)+1)
        self.compute_core_perfs(app)
        self.add_target(self.perf_target)
        self.iter_through_design(d2perf, 0, n_designs, candidate, app)
        return d2perf

    def print_latex(self):
        self.sheet1.printLatex()
