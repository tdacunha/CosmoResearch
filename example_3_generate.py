# -*- coding: utf-8 -*-

"""
Generate data for example: flat degeneracy
"""

###############################################################################
# initial imports:
import os
import numpy as np
import copy
import pickle
import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import tensiometer.gaussian_tension as gaussian_tension
from scipy import optimize

import synthetic_probability
from tensorflow.keras.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau()]

###############################################################################
# initial settings:

# output folder:
out_folder = '/Users/TaraD/Desktop/'
#out_folder = './results/example_3/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# cache for training:
flow_cache = out_folder+'flow_cache/'
if not os.path.exists(flow_cache):
    os.mkdir(flow_cache)

# number of samples:
n_samples = 100000

# cache file:
cache_file = out_folder+'example_3_cache.plk'

###############################################################################
# define the pdf:


def log_pdf(theta, theta0=[0.0, -0.5], sigma0=0.5):
    x, y = theta
    x0, y0 = theta0
    r2 = (x-x0)**2 + 20.*(-(y-y0) + 2.*(x-x0)**2)**2
    return -0.5*r2/sigma0**2


# prior:
prior = [-1., 1.]

###############################################################################
# generate the samples:

# if the cache file exists load it, otherwise generate it:
if os.path.isfile(cache_file):

    # load cache from pickle:
    cache_results = pickle.load(open(cache_file, 'rb'))
    # overload attribute to export cache content:

    def __getattr__(name):
        if name in cache_results.keys():
            return cache_results[name]
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

else:

    # initialize empty cache:
    cache_results = {}

    # find maximum posterior:
    print('Finding maximum posterior')
    temp_res = optimize.differential_evolution(lambda x: -log_pdf(x), [(prior[0], prior[1]), (prior[0], prior[1])])
    log_max_p = -temp_res.fun
    max_P_x = temp_res.x

    cache_results['log_max_p'] = copy.deepcopy(log_max_p)
    cache_results['max_P_x'] = copy.deepcopy(max_P_x)

    # generate samples from the distribution:
    print('Generating samples')
    samples = []
    likelihood = []
    num_samples = 0
    while num_samples < n_samples:
        xtemp = (prior[1]-prior[0])*np.random.rand(2) + prior[0]
        log_like = log_pdf(xtemp) - log_max_p
        if np.random.binomial(1, np.exp(log_like)):
            samples.append(xtemp)
            likelihood.append(log_like)
            num_samples += 1
    samples = np.array(samples)
    likelihood = np.array(likelihood)
    posterior_chain = MCSamples(samples=samples,
                                loglikes=-likelihood,
                                names=['theta_1', 'theta_2'],
                                labels=['\\theta_1', '\\theta_2'],
                                ranges={'theta_1': [prior[0], prior[1]],
                                        'theta_2': [prior[0], prior[1]]},
                                sampler='uncorrelated',
                                label='posterior')
    cache_results['posterior_chain'] = copy.deepcopy(posterior_chain)

    # generating prior:
    prior_samples = []
    for _min, _max in zip([prior[0], prior[0]], [prior[1], prior[1]]):
        prior_samples.append(np.random.uniform(_min, _max, size=n_samples))
    prior_samples = np.array(prior_samples).T
    prior_chain = MCSamples(samples=prior_samples,
                            names=['theta_1', 'theta_2'],
                            labels=['\\theta_1', '\\theta_2'],
                            ranges={'theta_1': [prior[0], prior[1]],
                                    'theta_2': [prior[0], prior[1]]},
                            sampler='uncorrelated',
                            label='prior')
    cache_results['prior_chain'] = copy.deepcopy(prior_chain)

    # dump cache to file for later plotting:
    pickle.dump(cache_results, open(cache_file, 'wb'), protocol=pickle.HIGHEST_PROTOCOL)
    #
    print('Saved generated example in', cache_file)

###############################################################################
# define the flows:

# if cache exists load training:
if os.path.isfile(flow_cache+'posterior'+'_permutations.pickle'):
    print('yes')
    # load trained model:
    posterior_chain = cache_results['posterior_chain']
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
    prior_chain = cache_results['prior_chain']
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

# # Testing to see if local metric problem is also here:
#
# param_ranges = [[-1.,1.],[-1.,1.]]
# coarse_P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 20)
# coarse_P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 20)
# coarse_x, coarse_y = coarse_P1, coarse_P2
# coarse_X, coarse_Y = np.meshgrid(coarse_x, coarse_y)
#
# coords = np.array([coarse_X, coarse_Y], dtype=np.float32).reshape(2, -1).T
# print(coords)
# local_metrics = posterior_flow.metric(coords)
# print(local_metrics)
