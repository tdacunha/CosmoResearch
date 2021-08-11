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
###############################################################################
# initial settings:

# output folder:
out_folder = './results/example_1/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

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

# posterior distribution:
posterior_mean = chain.getMeans([chain.index[name] for name in param_names])
posterior_cov = chain.cov([chain.index[name] for name in param_names])
posterior_distribution = GaussianND(posterior_mean, posterior_cov,
                                    names=['theta_'+str(i+1) for i in range(len(param_names))],
                                    labels=['\\theta_{'+str(i+1)+'}' for i in range(len(param_names))],
                                    label='posterior')

###############################################################################
# test plot if called directly:
if __name__ == '__main__':

    # plot distribution:
    g = plots.get_subplot_plotter()
    g.triangle_plot([prior_distribution, posterior_distribution], filled=True)
    g.export(out_folder+'0_prior_posterior.pdf')

    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_distribution], filled=True)
    g.export(out_folder+'0_posterior.pdf')



analyze_2d_example.run_example_2d(chain, prior_chain, [param_names[0],param_names[1]], outroot = '/Users/TaraD/Downloads/')
