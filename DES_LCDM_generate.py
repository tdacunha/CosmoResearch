# -*- coding: utf-8 -*-

"""
Read in and cache the flows for DES in LCDM.
"""

###############################################################################
# initial imports:
import os
import numpy as np

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import analyze_2d_example
import importlib

import synthetic_probability
from tensorflow.keras.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau()]

###############################################################################
# initial settings:

# output folder:
out_folder = './results/example_des_lcdm/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# cache for training:
flow_cache = out_folder+'flow_cache/'
if not os.path.exists(flow_cache):
    os.mkdir(flow_cache)

###############################################################################
# define the pdf from the DES samples:

# load the chains (remove no burn in since the example chains have already been cleaned):
chains_dir = here+'/chains/'
# the DES chain:
settings = {'ignore_rows': 0, 'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.3}


prior_chain_name = chains_dir+'001_DESY1_3x2_prior'
posterior_chain_name = chains_dir+'001_DESY1_3x2'

param_names = ['omegam',
               'omegab',
               'H0',
               'sigma8',
               'ns',
               'DES_b1',
               'DES_b2',
               'DES_b3',
               'DES_b4',
               'DES_b5',
               'DES_m1',
               'DES_m2',
               'DES_m3',
               'DES_m4',
               'DES_AIA',
               'DES_alphaIA',
               'DES_DzL1',
               'DES_DzL2',
               'DES_DzL3',
               'DES_DzL4',
               'DES_DzL5',
               'DES_DzS1',
               'DES_DzS2',
               'DES_DzS3',
               'DES_DzS4',]
num_params = len(param_names)
n_maf = 2*num_params
hidden_units = [num_params*2]*2
batch_size = 8192
epochs = 100
steps_per_epoch = 128


# load chains:
prior_chain = getdist.mcsamples.loadMCSamples(file_root=prior_chain_name, no_cache=True, settings=settings)
posterior_chain = getdist.mcsamples.loadMCSamples(file_root=posterior_chain_name, no_cache=True, settings=settings)


# initialize prior flow:
prior_flow = synthetic_probability.DiffFlowCallback(prior_chain, prior_bijector='ranges',
                                                    param_names=param_names, feedback=1)
# train prior flow:
prior_flow.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

# initialize posterior flow:
posterior_flow = synthetic_probability.DiffFlowCallback(posterior_chain,
                                                        prior_bijector=prior_flow.bijector,
                                                        param_ranges=prior_flow.parameter_ranges, param_names=prior_flow.param_names,
                                                        feedback=1, learning_rate=0.01, n_maf=n_maf, hidden_units=hidden_units)
# train posterior flow:
posterior_flow.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)



pass
