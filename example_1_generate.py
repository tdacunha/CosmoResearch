# -*- coding: utf-8 -*-

"""
Generate data for example: Gaussian with mean and covariance.
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
out_folder = './results/example_1/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# cache for training:
flow_cache = out_folder+'flow_cache/'
if not os.path.exists(flow_cache):
    os.mkdir(flow_cache)

# number of samples:
n_samples = 100000

###############################################################################
# define the pdf from the DES samples:

# load the chains (remove no burn in since the example chains have already been cleaned):
chains_dir = here+'/tensiometer/test_chains/'
# the DES chain:
settings = {'ignore_rows':0, 'smooth_scale_1D':0.3, 'smooth_scale_2D':0.3}
chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'DES', no_cache=True, settings=settings)
# the prior chain:
prior_chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'prior', no_cache=True, settings=settings)

# select parameters:
param_names = ['omegam', 'sigma8']

# prior distribution
prior_mean = prior_chain.getMeans([prior_chain.index[name] for name in param_names])
prior_cov = prior_chain.cov([prior_chain.index[name] for name in param_names])
prior_distribution = GaussianND(prior_mean, prior_cov,
                                names=['theta_'+str(i+1) for i in range(len(param_names))],
                                labels=['\\theta_{'+str(i+1)+'}' for i in range(len(param_names))],
                                label='prior')
prior_chain = prior_distribution.MCSamples(n_samples, label='prior')

# posterior distribution:
posterior_mean = chain.getMeans([chain.index[name] for name in param_names])
posterior_cov = chain.cov([chain.index[name] for name in param_names])
posterior_distribution = GaussianND(posterior_mean, posterior_cov,
                                    names=['theta_'+str(i+1) for i in range(len(param_names))],
                                    labels=['\\theta_{'+str(i+1)+'}' for i in range(len(param_names))],
                                    label='posterior')
posterior_chain = posterior_distribution.MCSamples(n_samples, label='posterior')


###############################################################################
# define the flows:

# if cache exists load training:
if os.path.isfile(flow_cache+'posterior'+'_permutations.pickle'):
    # load trained model:
    temp_MAF = synthetic_probability.SimpleMAF.load(len(posterior_chain.getParamNames().list()), flow_cache+'posterior')
    # initialize flow:
    posterior_flow = synthetic_probability.DiffFlowCallback(posterior_chain, Z2Y_bijector=temp_MAF.bijector, param_names=posterior_chain.getParamNames().list(), feedback=0, learning_rate=0.01)
else:
    # initialize flow:
    posterior_flow = synthetic_probability.DiffFlowCallback(posterior_chain, param_names=posterior_chain.getParamNames().list(), feedback=1, learning_rate=0.01)
    # train:
    posterior_flow.train(batch_size=8192, epochs=40, steps_per_epoch=128, callbacks=callbacks)
    # save trained model:
    posterior_flow.MAF.save(flow_cache+'posterior')

# if cache exists load training:
if os.path.isfile(flow_cache+'prior'+'_permutations.pickle'):
    # load trained model:
    temp_MAF = synthetic_probability.SimpleMAF.load(len(prior_chain.getParamNames().list()), flow_cache+'prior')
    # initialize flow:
    prior_flow = synthetic_probability.DiffFlowCallback(prior_chain, Z2Y_bijector=temp_MAF.bijector, param_names=prior_chain.getParamNames().list(), feedback=0, learning_rate=0.01)
else:
    # initialize flow:
    prior_flow = synthetic_probability.DiffFlowCallback(prior_chain, param_names=prior_chain.getParamNames().list(), feedback=1, learning_rate=0.01)
    # train:
    prior_flow.train(batch_size=8192, epochs=40, steps_per_epoch=128, callbacks=callbacks)
    # save trained model:
    prior_flow.MAF.save(flow_cache+'prior')


###############################################################################
# test plot if called directly:
if __name__ == '__main__':

    # feedback:
    print('* plotting generated sample')

    # plot distribution:
    g = plots.get_subplot_plotter()
    g.triangle_plot([prior_chain, posterior_chain], filled=True)
    g.export(out_folder+'0_prior_posterior.pdf')

    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain], filled=True)
    g.export(out_folder+'0_posterior.pdf')

    # plot learned posterior distribution:
    X_sample = np.array(posterior_flow.sample(n_samples))
    posterior_flow_chain = MCSamples(samples=X_sample,
                                     loglikes=-posterior_flow.log_probability(X_sample).numpy(),
                                     names=posterior_flow.param_names,
                                     label='Learned distribution')
    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain, posterior_flow_chain], filled=True)
    g.export(out_folder+'0_learned_posterior_distribution.pdf')

    # plot learned prior distribution:
    X_sample = np.array(prior_flow.sample(n_samples))
    prior_flow_chain = MCSamples(samples=X_sample,
                                 loglikes=-prior_flow.log_probability(X_sample).numpy(),
                                 names=prior_flow.param_names,
                                 label='Learned distribution')
    g = plots.get_subplot_plotter()
    g.triangle_plot([prior_chain, prior_flow_chain], filled=True)
    g.export(out_folder+'0_learned_prior_distribution.pdf')
