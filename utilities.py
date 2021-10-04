"""
Things that we will add to tensiometer utilities
"""

import numpy as np
from getdist import plots

def covariance_around(samples, center, weights=None):
    """
    Compute second moment around point
    """
    # number of samples and number of parameters:
    nsamps, npar = samples.shape
    # shift samples:
    diffs = samples - center
    diffs = diffs.T
    # initialize weights:
    if weights is None:
        weights = np.ones(nsamps)
    # do the calculation:
    cov = np.empty((npar, npar))
    for i, diff in enumerate(diffs):
        weightdiff = diff * weights
        for j in range(i, npar):
            cov[i, j] = weightdiff.dot(diffs[j])
            cov[j, i] = cov[i, j]
    cov /= np.sum(weights)
    #
    return cov
