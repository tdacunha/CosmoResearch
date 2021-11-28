# -*- coding: utf-8 -*-

"""
Generate data for example: Gaussian with mean and covariance.
"""

###############################################################################
# initial imports:
import os
import numpy as np
import pickle

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import analyze_2d_example
import importlib
import utilities

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
n_samples = 1000000

###############################################################################
# define the pdf from the DES samples:

# load the chains (remove no burn in since the example chains have already been cleaned):
chains_dir = here+'/tensiometer/test_chains/'
# the DES chain:
settings = {'ignore_rows': 0, 'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.3}
chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'DES', no_cache=True, settings=settings)
# the prior chain:
prior_chain_real = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'prior', no_cache=True, settings=settings)

# select parameters:
param_names = ['omegam', 'sigma8']

# add log of the chosen parameters:
for ch in [chain, prior_chain_real]:
    for name in param_names:
        ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log'+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()
param_names = ['log_'+name for name in param_names]

# posterior distribution:
posterior_mean = chain.getMeans([chain.index[name] for name in param_names])
posterior_cov = chain.cov([chain.index[name] for name in param_names])
posterior_distribution = GaussianND(posterior_mean, posterior_cov,
                                    names=['theta_'+str(i+1) for i in range(len(param_names))],
                                    labels=['\\theta_{'+str(i+1)+'}' for i in range(len(param_names))],
                                    label='posterior')
posterior_chain = posterior_distribution.MCSamples(n_samples, label='posterior')

# prior distribution
#prior_mean = posterior_mean
#prior_cov = utilities.covariance_around(prior_chain_real.samples[:, [prior_chain_real.index[name] for name in param_names]],
#                                        prior_mean, weights=prior_chain_real.weights)
prior_mean = prior_chain_real.getMeans([prior_chain_real.index[name] for name in param_names])
prior_cov = prior_chain_real.cov([prior_chain_real.index[name] for name in param_names])
prior_distribution = GaussianND(prior_mean, prior_cov,
                                names=['theta_'+str(i+1) for i in range(len(param_names))],
                                labels=['\\theta_{'+str(i+1)+'}' for i in range(len(param_names))],
                                label='prior')
prior_chain = prior_distribution.MCSamples(n_samples, label='prior')

###############################################################################
# define the flows:

# exact prior:
prior_bij = synthetic_probability.prior_bijector_helper(loc=prior_mean.astype(synthetic_probability.np_prec), cov=prior_cov.astype(synthetic_probability.np_prec))
prior_flow = synthetic_probability.DiffFlowCallback(prior_chain, prior_bijector=prior_bij, apply_pregauss=False, trainable_bijector=None, param_names=prior_chain.getParamNames().list(), feedback=1)

# posterior:
num_params = len(param_names)
n_maf = 3*num_params
hidden_units = [num_params*2]*3
batch_size = 2*8192
epochs = 80
steps_per_epoch = 128

# if cache exists load training:
if os.path.isfile(flow_cache+'posterior'+'_permutations.pickle'):
    # load trained model:
    temp_MAF = synthetic_probability.SimpleMAF.load(len(posterior_chain.getParamNames().list()), flow_cache+'posterior', n_maf=n_maf, hidden_units=hidden_units)
    # initialize flow:
    posterior_flow = synthetic_probability.DiffFlowCallback(posterior_chain, prior_bijector=prior_bij, trainable_bijector=temp_MAF.bijector, param_names=posterior_chain.getParamNames().list(), feedback=0, learning_rate=0.01)
else:
    # initialize flow:
    posterior_flow = synthetic_probability.DiffFlowCallback(posterior_chain, prior_bijector=prior_bij, param_names=posterior_chain.getParamNames().list(), feedback=1, learning_rate=0.01, n_maf=n_maf, hidden_units=hidden_units)
    # train:
    posterior_flow.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
    # save trained model:
    posterior_flow.MAF.save(flow_cache+'posterior')

# find MAP or load if it exists:
if os.path.isfile(flow_cache+'/posterior_MAP.pickle'):
    temp = pickle.load(open(flow_cache+'/posterior_MAP.pickle', 'rb'))
    posterior_flow.MAP_coord = temp['MAP_coord']
    posterior_flow.MAP_logP = temp['MAP_logP']
else:
    # find map:
    res = posterior_flow.MAP_finder(disp=True)
    print(res)
    # store:
    temp = {
            'MAP_coord': posterior_flow.MAP_coord,
            'MAP_logP': posterior_flow.MAP_logP,
            }
    # save out:
    pickle.dump(temp, open(flow_cache+'/posterior_MAP.pickle', 'wb'))

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
    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain, posterior_flow.MCSamples(n_samples)], filled=True, markers=posterior_flow.MAP_coord)
    g.export(out_folder+'0_learned_posterior_distribution.pdf')

    # plot learned prior distribution:
    g = plots.get_subplot_plotter()
    g.triangle_plot([prior_chain, prior_flow.MCSamples(n_samples)], filled=True)
    g.export(out_folder+'0_learned_prior_distribution.pdf')
