# -*- coding: utf-8 -*-

###############################################################################
# initial imports:

import os
import numpy as np
import DES_generate
import getdist
from getdist import plots
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import DES_generate

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
# import the tensiometer tools that we need:
from tensiometer import utilities
import synthetic_probability
from tensorflow_probability import bijectors as bj

###############################################################################
# initial settings:

# output folder:
out_folder = './results/evidence_playground/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

prior_chain = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'002_DESY1_shear_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'002_DESY1_shear', no_cache=True, settings=DES_generate.settings)

param_names = ['omegam', 'omegab', 'H0', 'sigma8', 'ns', 'DES_m1', 'DES_m2', 'DES_m3', 'DES_m4', 'DES_AIA', 'DES_alphaIA', 'DES_DzS1', 'DES_DzS2', 'DES_DzS3', 'DES_DzS4']
params_flow_cache = out_folder+'params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain, posterior_chain, params_flow_cache, pop_size=1)
params_prior_flow, params_posterior_flow = temp


params_posterior_flow.evidence()



samples = posterior_chain.samples[:, [posterior_chain.index[name] for name in param_names]]
flow_log_likes = params_posterior_flow.log_probability(params_posterior_flow.cast(samples))

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    average = np.average(values, weights=weights)
    # Fast and numerically precise:
    variance = np.average((values-average)**2, weights=weights)
    return (average, np.sqrt(variance))

np.mean(-posterior_chain.loglikes -flow_log_likes)
np.sqrt(np.var(-posterior_chain.loglikes -flow_log_likes))
weighted_avg_and_std(-posterior_chain.loglikes -flow_log_likes, posterior_chain.weights)


pass
