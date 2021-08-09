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

###############################################################################
# initial settings:

# output folder:
out_folder = './results/example_3/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# number of samples:
n_samples = 100000

# cache file:
cache_file = out_folder+'example_3_cache.plk'

###############################################################################
# define the pdf:


def log_pdf(theta, theta0=[0., 0.], rsigma=0.1):
    x, y = theta
    x0, y0 = theta0
    r = (x-x0) - (y-y0)**3
    return -0.5*r**2/rsigma**2


# prior:
prior = [-1., 1.]

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

    # plot distribution:
    g = plots.get_subplot_plotter()
    g.triangle_plot([prior_chain, posterior_chain], filled=True)
    g.export(out_folder+'0_prior_posterior.pdf')

    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain], filled=True)
    g.export(out_folder+'0_posterior.pdf')



pass
