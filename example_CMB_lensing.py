# -*- coding: utf-8 -*-

"""
Training of all DES Y3 examples
"""

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
out_folder = './results/example_CMB_lensing/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

###############################################################################
# import chains:

prior_chain_lcdm_CMB_lensing = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'008_CMB_lensing_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_lcdm_CMB_lensing = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'008_CMB_lensing', no_cache=True, settings=DES_generate.settings)

prior_chain_lcdm_CMB_lensing_shear = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'002_DESY1_shear_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_lcdm_CMB_lensing_shear = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'009_DESY1_shear_CMB_lensing', no_cache=True, settings=DES_generate.settings)

prior_chain_lcdm_CMB_lensing_3x2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'001_DESY1_3x2_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_lcdm_CMB_lensing_3x2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'009_DESY1_3x2_CMB_lensing', no_cache=True, settings=DES_generate.settings)

chains = [prior_chain_lcdm_CMB_lensing,
          posterior_chain_lcdm_CMB_lensing,
          prior_chain_lcdm_CMB_lensing_shear,
          posterior_chain_lcdm_CMB_lensing_shear,
          prior_chain_lcdm_CMB_lensing_3x2,
          posterior_chain_lcdm_CMB_lensing_3x2,
          ]

###############################################################################
# process chains:

# add log parameters:
for ch in chains:
    temp_names = ch.getParamNames().list()
    for name in temp_names:
        if np.all(ch.samples[:, ch.index[name]] > 0.):
            ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log '+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()


# identity bijector:
class identity_bj(bj.Identity):
    @property
    def name(self):
        return ''


###############################################################################
# train LCDM CMB lensing:

param_names = ['omegam', 'sigma8']
params_flow_cache = out_folder+'lcdm_CMB_lensing_2params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing, posterior_chain_lcdm_CMB_lensing, params_flow_cache)
lcdm_CMB_lensing_2params_prior_flow, lcdm_CMB_lensing_2params_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_log_2params_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_2params_prior_flow, transformation)
lcdm_CMB_lensing_log_2params_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_2params_posterior_flow, transformation)

param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns']
params_flow_cache = out_folder+'lcdm_CMB_lensing_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing, posterior_chain_lcdm_CMB_lensing, params_flow_cache)
lcdm_CMB_lensing_params_prior_flow, lcdm_CMB_lensing_params_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_log_params_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_params_prior_flow, transformation)
lcdm_CMB_lensing_log_params_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_params_posterior_flow, transformation)

###############################################################################
# train CMB lensing plus shear:

param_names = ['omegam', 'sigma8']
params_flow_cache = out_folder+'lcdm_CMB_lensing_shear_2params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing_shear, posterior_chain_lcdm_CMB_lensing_shear, params_flow_cache)
lcdm_CMB_lensing_shear_2params_prior_flow, lcdm_CMB_lensing_shear_2params_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_shear_log_2params_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_shear_2params_prior_flow, transformation)
lcdm_CMB_lensing_shear_log_2params_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_shear_2params_posterior_flow, transformation)

param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns']
params_flow_cache = out_folder+'lcdm_CMB_lensing_shear_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing_shear, posterior_chain_lcdm_CMB_lensing_shear, params_flow_cache)
lcdm_CMB_lensing_shear_params_prior_flow, lcdm_CMB_lensing_shear_params_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_shear_log_params_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_shear_params_prior_flow, transformation)
lcdm_CMB_lensing_shear_log_params_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_shear_params_posterior_flow, transformation)

param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns', 'DES_AIA', 'DES_alphaIA']
params_flow_cache = out_folder+'lcdm_CMB_lensing_shear_params_full_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing_shear, posterior_chain_lcdm_CMB_lensing_shear, params_flow_cache)
lcdm_CMB_lensing_shear_params_full_prior_flow, lcdm_CMB_lensing_shear_params_full_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_shear_log_params_full_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_shear_params_full_prior_flow, transformation)
lcdm_CMB_lensing_shear_log_params_full_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_shear_params_full_posterior_flow, transformation)

###############################################################################
# train CMB lensing plus 3x2:

param_names = ['omegam', 'sigma8']
params_flow_cache = out_folder+'lcdm_CMB_lensing_3x2_2params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing_3x2, posterior_chain_lcdm_CMB_lensing_3x2, params_flow_cache)
lcdm_CMB_lensing_3x2_2params_prior_flow, lcdm_CMB_lensing_3x2_2params_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_3x2_log_2params_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_3x2_2params_prior_flow, transformation)
lcdm_CMB_lensing_3x2_log_2params_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_3x2_2params_posterior_flow, transformation)

param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns']
params_flow_cache = out_folder+'lcdm_CMB_lensing_3x2_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing_3x2, posterior_chain_lcdm_CMB_lensing_3x2, params_flow_cache)
lcdm_CMB_lensing_3x2_params_prior_flow, lcdm_CMB_lensing_3x2_params_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_3x2_log_params_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_3x2_params_prior_flow, transformation)
lcdm_CMB_lensing_3x2_log_params_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_3x2_params_posterior_flow, transformation)

param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns', 'DES_b1', 'DES_b2', 'DES_b3', 'DES_b4', 'DES_b5', 'DES_AIA', 'DES_alphaIA']
params_flow_cache = out_folder+'lcdm_CMB_lensing_3x2_params_full_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing_3x2, posterior_chain_lcdm_CMB_lensing_3x2, params_flow_cache)
lcdm_CMB_lensing_3x2_params_full_prior_flow, lcdm_CMB_lensing_3x2_params_full_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_3x2_log_params_full_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_3x2_params_full_prior_flow, transformation)
lcdm_CMB_lensing_3x2_log_params_full_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_3x2_params_full_posterior_flow, transformation)
