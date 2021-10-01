# -*- coding: utf-8 -*-

"""
LCDM DES example
"""

###############################################################################
# initial imports:
import os
import numpy as np
import DES_generate
import getdist
from getdist import plots, MCSamples

###############################################################################
# initial settings:

# output folder:
out_folder = './results/example_DES_1/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

###############################################################################
# import chains:

prior_chain = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'001_DESY1_3x2_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'001_DESY1_3x2', no_cache=True, settings=DES_generate.settings)

###############################################################################
# process chains:

# add log parameters:
for ch in [prior_chain, posterior_chain]:
    temp_names = ch.getParamNames().list()
    for name in temp_names:
        if np.all(ch.samples[:, ch.index[name]] > 0.):
            ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log'+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()

###############################################################################
# sanity triangle plot:

param_names = ['omegam', 'omegab', 'H0', 'sigma8', 'ns',
               'DES_b1', 'DES_b2', 'DES_b3', 'DES_b4', 'DES_b5',
               'DES_m1', 'DES_m2', 'DES_m3', 'DES_m4',
               'DES_AIA', 'DES_alphaIA',
               'DES_DzL1', 'DES_DzL2', 'DES_DzL3', 'DES_DzL4', 'DES_DzL5',
               'DES_DzS1', 'DES_DzS2', 'DES_DzS3', 'DES_DzS4']
g = plots.get_subplot_plotter()
g.triangle_plot([prior_chain, posterior_chain], params=param_names, filled=False)
g.export(out_folder+'/0_prior_posterior_distribution.pdf')

###############################################################################
# PCA in log space:

param_names = ['log_omegam', 'log_omegab', 'log_H0', 'log_sigma8', 'ns',
               'log_DES_b1', 'log_DES_b2', 'log_DES_b3', 'log_DES_b4', 'log_DES_b5',
               'DES_m1', 'DES_m2', 'DES_m3', 'DES_m4',
               'DES_AIA', 'DES_alphaIA',
               'DES_DzL1', 'DES_DzL2', 'DES_DzL3', 'DES_DzL4', 'DES_DzL5',
               'DES_DzS1', 'DES_DzS2', 'DES_DzS3', 'DES_DzS4']
covariance = posterior_chain.cov(param_names)
eig, eigv = np.linalg.eigh(covariance)


###############################################################################
# KL in log space:

param_names = ['log_omegam', 'log_omegab', 'log_H0', 'log_sigma8', 'ns',
               'log_DES_b1', 'log_DES_b2', 'log_DES_b3', 'log_DES_b4', 'log_DES_b5',
               'DES_m1', 'DES_m2', 'DES_m3', 'DES_m4',
               'DES_AIA', 'DES_alphaIA',
               'DES_DzL1', 'DES_DzL2', 'DES_DzL3', 'DES_DzL4', 'DES_DzL5',
               'DES_DzS1', 'DES_DzS2', 'DES_DzS3', 'DES_DzS4']
covariance = posterior_chain.cov(param_names)
prior_covariance = prior_chain.cov(param_names)






pass
