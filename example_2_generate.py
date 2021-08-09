# -*- coding: utf-8 -*-

"""
Generate data for example: non-Gaussian resembling DES.
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
import tensiometer.gaussian_tension as gaussian_tension

###############################################################################
# initial settings:

# output folder:
out_folder = './results/example_2/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

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

## get the posterior chain:

# add log of the parameters:
for ch in [chain]:
    for name in param_names:
        ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log'+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()

# now we generate the Gaussian approximation:
temp_param_names = ['log_'+name for name in param_names]

# create the Gaussian:
gauss = gaussian_tension.gaussian_approximation(chain, param_names=temp_param_names)
posterior_chain = gauss.MCSamples(size=n_samples)

# exponentiate:
for ch in [posterior_chain]:
    p = ch.getParams()
    # the original parameters:
    for ind, name in enumerate(temp_param_names):
        ch.addDerived(np.exp(ch.samples[:, ch.index[name]]), name=str(name).replace('log_',''), label=str(ch.getParamNames().parWithName(name).label).replace('\\log',''))
    # label=ch.getParamNames().parWithName(name).label.replace('\\log ','')
    ch.updateBaseStatistics()

# define the new samples:
posterior_chain = MCSamples(samples=posterior_chain.samples[:, [posterior_chain.index[name] for name in param_names]],
                            names=['theta_'+str(i+1) for i in range(len(param_names))],
                            labels=['\\theta_{'+str(i+1)+'}' for i in range(len(param_names))],
                            label='posterior')

## get the prior chain:

_mins = np.amin(prior_chain.samples[:, [prior_chain.index[name] for name in param_names]], axis=0)
_maxs = np.amax(prior_chain.samples[:, [prior_chain.index[name] for name in param_names]], axis=0)

_mins = [0., 0.]
_maxs = [1., 2.]

prior_samples = []
for _min, _max in zip(_mins, _maxs):
    prior_samples.append(np.random.uniform(_min, _max, size=n_samples))
prior_samples = np.array(prior_samples).T

prior_chain = MCSamples(samples=prior_samples,
                        names=['theta_'+str(i+1) for i in range(len(param_names))],
                        labels=['\\theta_{'+str(i+1)+'}' for i in range(len(param_names))],
                        label='prior')

###############################################################################
# test plot if called directly:
if __name__ == '__main__':

    # plot distribution:
    g = plots.get_subplot_plotter()
    g.triangle_plot([prior_chain, posterior_chain], filled=True)
    g.export(out_folder+'0_prior_posterior.pdf')

    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain], filled=True)
    g.export(out_folder+'0_posterior.pdf')
