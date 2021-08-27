"""
General code to analyze examples

For testing purposes:

import example_3_generate as example
import example_2_generate as example
import example_5_generate as example
import example_1_generate as example


chain = example.posterior_chain
prior_chain = example.prior_chain
flow = example.posterior_flow
param_names = example.posterior_chain.getParamNames().list()
outroot = example.out_folder
train_params = {}
param_ranges = None #[[-1.5, 1.5], [-1.5, 1.5]] #None # [[0.0, 0.6], [0.4, 1.5]]

# for testing prior run
chain = example.prior_chain
prior_chain = example.prior_chain
flow = example.prior_flow
param_names = example.prior_chain.getParamNames().list()
outroot = example.out_folder
train_params = {}
param_ranges=[[0.0, 0.7], [0, 1.7]]# None #[[-1.5, 1.5], [-1.5, 1.5]] #None # [[0.0, 0.6], [0.4, 1.5]]
"""

###############################################################################
# initial imports:
import os
import numpy as np

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import synthetic_probability
from tensorflow.keras.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau()]
from getdist import plots, MCSamples

from scipy import optimize
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import scipy.stats
import matplotlib.gridspec as gridspec

# import the tensiometer tools that we need:
from tensiometer import utilities
from tensiometer import gaussian_tension
from tensiometer import mcmc_tension

# tensorflow imports:
import tensorflow as tf
import tensorflow_probability as tfp

###############################################################################
# general utility functions:


# helper function to get levels:
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
        try:
            res = optimize.brentq(lambda x: _helper(x)-c, np.amin(P), np.amax(P))
            levs.append(res)
        except:
            print('Cannot generate proper levels')
            levs = len(conf)
            break

    return levs


def run_KL_example_2d(chain, prior_chain, flow, prior_flow, param_names, outroot, param_ranges=None, train_params={}):
    """
    Run full KL analysis of 2d example case, as in flow playground
    """

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

    log_Pi = prior_flow.log_probability(np.array([X, Y], dtype=np.float32).T)
    log_Pi = np.array(log_Pi).T
    Pi = np.exp(log_Pi)
    Pi = Pi / simps(simps(Pi, y), x)

    log_L = log_P - log_Pi
    Like = np.exp(log_L)
    Like = Like / simps(simps(Like, y), x)

    # find the MAPS:
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
    # TF KL decomposition:

    @tf.function
    def tf_KL_decomposition(matrix_a, matrix_b):
        """
        """
        # compute the eigenvalues of b, lambda_b:
        _lambda_b, _phi_b = tf.linalg.eigh(matrix_b)
        _sqrt_lambda_b = tf.linalg.diag(1./tf.math.sqrt(_lambda_b))
        _phib_prime = tf.matmul(_phi_b, _sqrt_lambda_b)
        #
        trailing_axes = [-1, -2]
        leading = tf.range(tf.rank(_phib_prime) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(_phib_prime)
        new_order = tf.concat([leading, trailing], axis=0)
        _phib_prime_T = tf.transpose(_phib_prime, new_order)
        #
        _a_prime = tf.matmul(tf.matmul(_phib_prime_T, matrix_a), _phib_prime)
        _lambda, _phi_a = tf.linalg.eigh(_a_prime)
        _phi = tf.matmul(tf.matmul(_phi_b, _sqrt_lambda_b), _phi_a)
        return _lambda, _phi

    ###############################################################################
    # Plot of KL eigenvectors at mean
    ###############################################################################

    alpha = np.linspace(-1, 1, 1000)
    plt.figure(figsize=figsize)

    # plot KL of flow covariance metric
    eig, eigv = tf_KL_decomposition(prior_covariance_metric, covariance_metric)
    eig, eigv = eig.numpy(), eigv.numpy()
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
    eig, eigv = tf_KL_decomposition(prior_fisher_metric, fisher_metric)
    eig, eigv = eig.numpy(), eigv.numpy()
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
    eig, eigv = tf_KL_decomposition(prior_cov_samples, cov_samples)
    eig, eigv = eig.numpy(), eigv.numpy()
    inds = (np.argsort(eig)[::-1])
    param_directions = np.linalg.inv(eigv.T)
    eigv = (param_directions.T[inds]).T
    mode = 0
    norm0 = np.linalg.norm(eigv[:,0])
    plt.plot(mean[0]+alpha*eigv[0, mode]/norm0, mean[1]+alpha*eigv[1, mode]/norm0, lw=1., color='red', ls='--', label='KL samples',alpha = .5)
    mode = 1
    norm1 = np.linalg.norm(eigv[:,1])
    plt.plot(mean[0]+alpha*eigv[0, mode]/norm1, mean[1]+alpha*eigv[1, mode]/norm1, lw=1., color='red', ls='--',alpha = .5)

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
