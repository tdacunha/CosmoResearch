# import libraries:
import sys, os
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)

from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import getdist
getdist.chains.print_load_details = False
import scipy
import matplotlib.pyplot as plt
import IPython
from IPython.display import Markdown
import numpy as np
import seaborn as sns
# import the tensiometer tools that we need:
from tensiometer import utilities
from tensiometer import gaussian_tension
from tensiometer import mcmc_tension

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from scipy import optimize
from scipy.integrate import simps

import synthetic_probability
import importlib
importlib.reload(synthetic_probability)

import copy
import pickle

from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import tensiometer.gaussian_tension as gaussian_tension
from scipy import optimize

import example_2_generate as example

chain=example.posterior_chain
flow=example.posterior_flow
param_names=example.posterior_chain.getParamNames().list()
param_ranges=[[0.0, 0.6], [0.4, 1.5]]
outroot=example.out_folder+'posterior_'

prior_chain=example.prior_chain
prior_flow=example.prior_flow
prior_param_names=example.prior_chain.getParamNames().list()
prior_outroot=example.out_folder+'prior_'

def get_levels(P, x, y, conf=[0.95, 0.68]):
    """
    Get levels from a 2D grid
    """
    def _helper(alpha):
        _P = P.copy()
        _P[_P < alpha] = 0.
        return simps(simps(_P, y), x)
    levs = []
    for c in conf:
        res = optimize.brentq(lambda x: _helper(x)-c, np.amin(P), np.amax(P))
        levs.append(res)
    return levs

# plotting preferences:
figsize = (8, 8)
fontsize = 15

# parameter ranges for plotting from the prior:
if param_ranges is None:
    param_ranges = np.array([np.amin(chain.samples, axis=0), np.amax(chain.samples, axis=0)]).T
    param_ranges = param_ranges[[chain.index[name] for name in param_names], :]

# parameter labels:
param_labels = [name.label for name in chain.getParamNames().parsWithNames(param_names)]
param_labels_latex = ['$'+name+'$' for name in param_labels]

# parameter grids:
P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 200)
P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 200)
x, y = P1, P2
X, Y = np.meshgrid(x, y)

# coarse parameter grid:
coarse_P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 20)
coarse_P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 20)
coarse_x, coarse_y = coarse_P1, coarse_P2
coarse_X, coarse_Y = np.meshgrid(coarse_x, coarse_y)

# obtain posterior on grid from samples:
density = chain.get2DDensity(param_names[0], param_names[1], normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
density.P = density.P / simps(simps(density.P, density.y), density.x)

# levels for contour plots:
levels_5 = [utilities.from_sigma_to_confidence(i) for i in range(5, 1, -1)]
levels_3 = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]

log_P = flow.log_probability(np.array([X, Y], dtype=np.float32).T)
log_P = np.array(log_P).T
P = np.exp(log_P)
P = P / simps(simps(P, y), x)

# find the MAP in parameter space:
result = flow.MAP_finder(disp=True)
maximum_posterior = result.x
# mean:
mean = chain.getMeans([chain.index[name] for name in param_names])

cov_samples = chain.cov(pars=param_names)
prior_cov_samples = prior_chain.cov(pars=param_names)


# metrics from flow around mean:
covariance_metric = flow.metric(np.array([mean]).astype(np.float32))[0]
fisher_metric = flow.inverse_metric(np.array([mean]).astype(np.float32))[0]

prior_covariance_metric = prior_flow.metric(np.array([mean]).astype(np.float32))[0]
prior_fisher_metric = prior_flow.inverse_metric(np.array([mean]).astype(np.float32))[0]


###############################################################################

alpha = np.linspace(-1, 1, 1000)
plt.figure(figsize=figsize)

# plot KL of flow covariance metric
eig, eigv = utilities.KL_decomposition(prior_covariance_metric, covariance_metric)
print(eigv)
# inds = (np.argsort(eig)[::-1])
# param_directions = np.linalg.inv(eigv.T)
# eigv = (param_directions.T[inds]).T
print(eigv)
mode = 0
norm0 = np.linalg.norm(eigv[:,0])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm0, mean[1]+alpha*eigv[1, mode]/norm0, lw=1.5, color='k', ls='--', label='KL flow covariance', alpha = .5)
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm1, mean[1]+alpha*eigv[1, mode]/norm1, lw=1.5, color='k', ls='--',alpha = .5)

# plot KL of flow fisher metric
eig, eigv = utilities.KL_decomposition(prior_fisher_metric, fisher_metric)
inds = (np.argsort(eig)[::-1])
param_directions = np.linalg.inv(eigv.T)
eigv = (param_directions.T[inds]).T

mode = 0
norm0 = np.linalg.norm(eigv[:,0])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm0, mean[1]+alpha*eigv[1, mode]/norm0, lw=1., color='green', ls='-.', label='KL flow fisher',alpha = .5)
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm1, mean[1]+alpha*eigv[1, mode]/norm1, lw=1., color='green', ls='-.',alpha = .5)

# plot KL of covariance of samples
eig, eigv = utilities.KL_decomposition(prior_cov_samples, cov_samples)
inds = (np.argsort(eig)[::-1])
param_directions = np.linalg.inv(eigv.T)
eigv = (param_directions.T[inds]).T
mode = 0
norm0 = np.linalg.norm(eigv[:,0])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm0, mean[1]+alpha*eigv[1, mode]/norm0, lw=1., color='red', ls='--', label='KL samples',alpha = .5)
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm1, mean[1]+alpha*eigv[1, mode]/norm1, lw=1., color='red', ls='--',alpha = .5)

# plot PCA of flow covariance metric
_, eigv = np.linalg.eigh(covariance_metric)
mode = 0
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1.5, color='k', ls='--', label='PCA flow covariance')
mode = 1
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1.5, color='k', ls='--')

# plot PCA of flow fisher metric
_, eigv = np.linalg.eigh(fisher_metric)
mode = 0
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='green', ls='-.', label='PCA flow fisher')
mode = 1
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='green', ls='-.')

# # plot PCA of covariance of samples
# _, eigv = np.linalg.eigh(cov_samples)
# mode = 0
# plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='red', ls='--', label='samples')
# mode = 1
# plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='red', ls='--')

# plot contours
plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
plt.scatter(mean[0], mean[1], color='k')

plt.legend()
plt.xlim([np.amin(P1), np.amax(P1)])
plt.ylim([np.amin(P2), np.amax(P2)])
plt.legend()
plt.xlabel(param_labels_latex[0], fontsize=fontsize)
plt.ylabel(param_labels_latex[1], fontsize=fontsize)
plt.tight_layout()
plt.show()
