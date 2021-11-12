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
out_folder = './results/example_DES_Y3/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

###############################################################################
# import chains:

prior_chain_lcdm_shear = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'002_DESY1_shear_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_lcdm_shear = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'005_DESY3_shear_fake_check', no_cache=True, settings=DES_generate.settings)

prior_chain_wcdm_shear = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'004_DESY1_shear_wCDM_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_wcdm_shear = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'006_DESY3_shear_fake_wCDM', no_cache=True, settings=DES_generate.settings)

prior_chain_mnu_shear = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'004_DESY1_shear_mnu_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_mnu_shear = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'007_DESY3_shear_fake_mnu', no_cache=True, settings=DES_generate.settings)

prior_chain_lcdm_3x2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'001_DESY1_3x2_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_lcdm_3x2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'005_DESY3_3x2_fake_check', no_cache=True, settings=DES_generate.settings)

prior_chain_wcdm_3x2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'003_DESY1_3x2_wCDM_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_wcdm_3x2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'006_DESY3_3x2_fake_wCDM', no_cache=True, settings=DES_generate.settings)

prior_chain_mnu_3x2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'003_DESY1_3x2_mnu_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_mnu_3x2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'007_DESY3_3x2_fake_mnu', no_cache=True, settings=DES_generate.settings)

chains = [prior_chain_lcdm_shear,
          posterior_chain_lcdm_shear,
          prior_chain_wcdm_shear,
          posterior_chain_wcdm_shear,
          prior_chain_mnu_shear,
          posterior_chain_mnu_shear,
          prior_chain_lcdm_3x2,
          posterior_chain_lcdm_3x2,
          prior_chain_wcdm_3x2,
          posterior_chain_wcdm_3x2,
          prior_chain_mnu_3x2,
          posterior_chain_mnu_3x2]

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

###############################################################################
# train shear LCDM:

log_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'log_ns']
log_params_flow_cache = out_folder+'lcdm_shear_log_params_flow_cache'
temp = DES_generate.helper_load_chains(log_param_names, prior_chain_lcdm_shear, posterior_chain_lcdm_shear, log_params_flow_cache)
lcdm_shear_log_params_prior_flow, lcdm_shear_log_params_posterior_flow = temp
#
param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns']
params_flow_cache = out_folder+'lcdm_shear_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_shear, posterior_chain_lcdm_shear, params_flow_cache)
lcdm_shear_params_prior_flow, lcdm_shear_params_posterior_flow = temp

###############################################################################
# train shear wCDM:

log_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'log_ns', 'w']
log_params_flow_cache = out_folder+'wcdm_shear_log_params_flow_cache'
temp = DES_generate.helper_load_chains(log_param_names, prior_chain_wcdm_shear, posterior_chain_wcdm_shear, log_params_flow_cache)
log_params_prior_flow, log_params_posterior_flow = temp
#
param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns', 'w']
params_flow_cache = out_folder+'wcdm_shear_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_wcdm_shear, posterior_chain_wcdm_shear, params_flow_cache)
params_prior_flow, params_posterior_flow = temp

###############################################################################
# train shear mnu:

log_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'log_ns', 'log_mnu']
log_params_flow_cache = out_folder+'mnu_shear_log_params_flow_cache'
temp = DES_generate.helper_load_chains(log_param_names, prior_chain_mnu_shear, posterior_chain_mnu_shear, log_params_flow_cache)
log_params_prior_flow, log_params_posterior_flow = temp
#
param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns', 'mnu']
params_flow_cache = out_folder+'mnu_shear_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_mnu_shear, posterior_chain_mnu_shear, params_flow_cache)
params_prior_flow, params_posterior_flow = temp

###############################################################################
# train 3x2 LCDM:

log_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'log_ns']
log_params_flow_cache = out_folder+'lcdm_3x2_log_params_flow_cache'
temp = DES_generate.helper_load_chains(log_param_names, prior_chain_lcdm_3x2, posterior_chain_lcdm_3x2, log_params_flow_cache)
log_params_prior_flow, log_params_posterior_flow = temp
#
param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns']
params_flow_cache = out_folder+'lcdm_3x2_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_3x2, posterior_chain_lcdm_3x2, params_flow_cache)
params_prior_flow, params_posterior_flow = temp

###############################################################################
# train 3x2 wCDM:

log_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'log_ns', 'w']
log_params_flow_cache = out_folder+'wcdm_3x2_log_params_flow_cache'
temp = DES_generate.helper_load_chains(log_param_names, prior_chain_wcdm_3x2, posterior_chain_wcdm_3x2, log_params_flow_cache)
log_params_prior_flow, log_params_posterior_flow = temp
#
param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns', 'w']
params_flow_cache = out_folder+'wcdm_3x2_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_wcdm_3x2, posterior_chain_wcdm_3x2, params_flow_cache)
params_prior_flow, params_posterior_flow = temp

###############################################################################
# train 3x2 mnu:

log_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'log_ns', 'log_mnu']
log_params_flow_cache = out_folder+'mnu_3x2_log_params_flow_cache'
temp = DES_generate.helper_load_chains(log_param_names, prior_chain_mnu_3x2, posterior_chain_mnu_3x2, log_params_flow_cache)
log_params_prior_flow, log_params_posterior_flow = temp
#
param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns', 'mnu']
params_flow_cache = out_folder+'mnu_3x2_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_mnu_3x2, posterior_chain_mnu_3x2, params_flow_cache)
params_prior_flow, params_posterior_flow = temp
