import numpy as np
import logging
import mcerp
import functools

from mcerp import *
from scipy.stats import norm
from models.custom_distributions import CustomDistribution
from utils.boxcox import BoxCox
from utils.plotting import PlotHelper

#mcerp.npts = 20

# Hidden distribution.
X = LogN(0, .25)
#X = Binomial(1000, .9)/1000
print 'X mean: {}'.format(np.mean(X._mcpts))
shift = np.amin(X._mcpts) - CustomDistribution.NON_ZERO_FACTOR
if shift < 0:
    X_samples = X._mcpts - shift
else:
    X_samples = X._mcpts

# Box-Cox transformation.
a = BoxCox.find_lambda(X_samples)
print 'Alpha: {}'.format(a)
Y_samples = BoxCox.transform(X_samples, a)
Y_mean = np.mean(Y_samples)
Y_min = np.amin(Y_samples)
Y_max = np.amax(Y_samples)
# Fit in transformed domain.
loc, scale = norm.fit(Y_samples)
xs = np.arange(Y_min, Y_max, .0005)

# Resample.
bc_lower, bc_upper = None, None
if a > 0:
    bc_lower = -1. / a
    bc_upper = 2 * loc - bc_lower
if a < 0:
    bc_upper = -1. / a
    bc_lower = 2 * loc - bc_upper
#mcerp.npts = 10000
Z = CustomDistribution.GaussianDistribution(loc, scale, bc_lower, bc_upper)

# Back-transform.
Xp_samples = BoxCox.back_transform(Z._mcpts, a)

PlotHelper.plot_hist(X_samples, title='original')
PlotHelper.plot_overlap_hists([X_samples, Xp_samples], title='original_resample')
print 'X_samples mean: {}'.format(np.mean(X_samples))
print 'Xp_samples mean: {}'.format(np.mean(Xp_samples))
PlotHelper.plot_KDE(xs, functools.partial(norm.pdf, loc=loc, scale=scale), ground_truth = Y_samples, title='transformation')
