# -*- coding: utf-8 -*-

"""
General code to analyze examples

For testing purposes:

import example_1_generate as example

chain=example.posterior_chain
flow=example.posterior_flow
param_names=example.posterior_chain.getParamNames().list()
param_ranges=[[0.0, 0.5], [0.3, 1.5]]

outroot=example.out_folder+'posterior_'
use_MAP = True

# for testing prior run
chain=example.prior_chain
flow=example.prior_flow
param_names=example.prior_chain.getParamNames().list()
param_ranges=[[0.01, 0.7-0.01], [0.01, 1.7-0.01]]
outroot=example.out_folder+'prior_'
use_MAP=False
"""

###############################################################################
# initial imports:
import os
import numpy as np
import copy
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
    levs = np.sort(levs)
    return levs


###############################################################################

def run_example_2d(chain, flow, param_names, outroot, param_ranges=None, use_MAP=True):
    """
    Run full analysis of 2d example case, as in flow playground
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

    ###########################################################################
    # Plot learned distribution from samples:
    ###########################################################################

    # feedback:
    print('1) learned posterior')

    N = 10000
    X_sample = np.array(flow.sample(N))
    flow_chain = MCSamples(samples=X_sample,
                           loglikes=-flow.log_probability(X_sample).numpy(),
                           names=param_names,
                           label='Learned distribution')

    g = plots.get_subplot_plotter()
    g.triangle_plot([chain, flow_chain], params=param_names, filled=False)
    g.export(outroot+'1_learned_posterior_distribution.pdf')
    plt.close('all')

    ###########################################################################
    # Plot learned distribution using flow log probability (and samples):
    ###########################################################################

    # feedback:
    print('2) learned posterior from samples')

    # compute flow probability on a grid:
    log_P = flow.log_probability(np.array([X, Y], dtype=np.float32).T)
    log_P = np.array(log_P).T
    P = np.exp(log_P)
    P = P / simps(simps(P, y), x)

    # plot learned contours
    plt.figure(figsize=figsize)
    plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels_5), linewidths=1., linestyles='--', colors=['red' for i in levels_5])
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'2_log_prob_distribution.pdf')
    plt.close('all')

    ###########################################################################
    # Plot samples:
    ###########################################################################

    # transfer samples in abstract space:
    abstract_samples = flow.map_to_abstract_coord(flow_chain.samples)

    # in parameter space:
    plt.figure(figsize=(2*figsize[0], figsize[1]))
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    ax1.scatter(flow_chain.samples[:, 0], flow_chain.samples[:, 1], s=0.3, c=flow_chain.loglikes)
    ax2.scatter(abstract_samples[:, 0], abstract_samples[:, 1], s=0.3, c=flow_chain.loglikes)
    ax1.set_xlabel(param_labels_latex[0], fontsize=fontsize)
    ax1.set_ylabel(param_labels_latex[1], fontsize=fontsize)
    ax2.set_xlabel('$Z_1$', fontsize=fontsize)
    ax2.set_ylabel('$Z_2$', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'2_learned_distribution_samples.pdf')
    plt.close('all')

    ###########################################################################
    # Plot log determinant of metric:
    ###########################################################################

    # feedback:
    print('3) log determinant')

    # compute log determinant of metric:
    log_det = flow.log_det_metric(np.array([X, Y], dtype=np.float32).T)
    log_det = np.array(log_det).T

    # plot meshgrid of log determinant
    plt.figure(figsize=figsize)
    pc = plt.pcolormesh(X, Y, log_det, linewidth=0, rasterized=True, shading='auto', cmap='RdBu')
    colorbar = plt.colorbar(pc)

    # plot contours
    plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'3_log_det_jacobian_distribution.pdf')
    plt.close('all')

    ###########################################################################
    # Plot maximum posterior and mean:
    ###########################################################################

    # feedback:
    print('4) MAP and mean')

    # find the MAP in parameter space:
    if hasattr(flow, 'MAP_coord'):
        maximum_posterior = flow.MAP_coord
    else:
        result = flow.MAP_finder(disp=True)
        maximum_posterior = result.x
    # mean:
    mean = chain.getMeans([chain.index[name] for name in param_names])

    if use_MAP:
        reference_point = maximum_posterior
    else:
        reference_point = mean

    # find in abstract space:
    maximum_posterior_abs = flow.map_to_abstract_coord(flow.cast(maximum_posterior))
    mean_abs = flow.map_to_abstract_coord(flow.cast(mean))

    plt.figure(figsize=(2*figsize[0], figsize[1]))
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])
    # plot contours with MAP and mean in parameter space:
    ax1.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    ax1.scatter(maximum_posterior[0], maximum_posterior[1], color='green', label='MAP: (%.3f, %.3f)' %(maximum_posterior[0],maximum_posterior[1]))
    ax1.scatter(mean[0], mean[1], color='red', label='mean: (%.3f, %.3f)' %(mean[0],mean[1]))
    ax1.legend()
    ax1.set_xlim([np.amin(P1), np.amax(P1)])
    ax1.set_ylim([np.amin(P2), np.amax(P2)])
    ax1.set_xlabel(param_labels_latex[0], fontsize=fontsize)
    ax1.set_ylabel(param_labels_latex[1], fontsize=fontsize)

    # plot contours with MAP and mean in abstract space:

    # print the iso-contours:
    origin = [0, 0]
    theta = np.linspace(0.0, 2.*np.pi, 200)
    for i in range(4):
        _length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(i), 2))
        ax2.plot(origin[0]+_length*np.sin(theta), origin[1]+_length*np.cos(theta), ls='-', lw=1., color='k')
    ax2.scatter(maximum_posterior_abs[0], maximum_posterior_abs[1], color='green', label='MAP: (%.3f, %.3f)' %(maximum_posterior_abs[0],maximum_posterior_abs[1]))
    ax2.scatter(mean_abs[0], mean_abs[1], color='red', label='mean: (%.3f, %.3f)' %(mean_abs[0], mean_abs[1]))
    ax2.legend()
    ax2.set_xlabel(param_labels_latex[0], fontsize=fontsize)
    ax2.set_ylabel(param_labels_latex[1], fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(outroot+'4_maximum_posterior_and_sample_mean.pdf')
    plt.close('all')

    ###########################################################################
    # Plot comparison of covariance metric, fisher metric,
    ###########################################################################

    # feedback:
    print('5) global covariance')

    # covariance from samples
    cov_samples = chain.cov(pars=param_names)

    # metrics from flow around mean:
    covariance_metric = flow.metric(flow.cast([mean]))[0]
    fisher_metric = flow.inverse_metric(flow.cast([mean]))[0]

    alpha = np.linspace(-1, 1, 1000)
    plt.figure(figsize=figsize)

    # plot PCA of flow covariance metric
    _, eigv = np.linalg.eigh(covariance_metric)
    mode = 0
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1.5, color='k', ls='--', label='flow covariance')
    mode = 1
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1.5, color='k', ls='--')

    # plot PCA of flow fisher metric
    _, eigv = np.linalg.eigh(fisher_metric)
    mode = 0
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='green', ls='-.', label='flow fisher')
    mode = 1
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='green', ls='-.')

    # plot PCA of covariance of samples
    _, eigv = np.linalg.eigh(cov_samples)
    mode = 0
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='red', ls='--', label='samples')
    mode = 1
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='red', ls='--')

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
    plt.savefig(outroot+'5_comparison_of_cov_fisher_samples_at_mean.pdf')
    plt.close('all')

    ###########################################################################
    # trace geodesics in abstract space passing from the maximum posterior
    ###########################################################################

    # feedback:
    print('6) geodesics')

    # find where the MAP goes:
    reference_image = flow.map_to_abstract_coord(flow.cast(reference_point))

    # compute geodesics aligned with abstract coordinate axes:
    length = flow.sigma_to_length(4.)
    r = np.linspace(-length, length, 1000)
    t = 0.0
    geo = np.array([reference_image[0] + r*np.cos(t), reference_image[1] + r*np.sin(t)], dtype=np.float32)
    geo_1 = flow.map_to_original_coord(geo.T)
    abs_geo_1 = copy.deepcopy(geo.T)
    t = np.pi/2.
    geo = np.array([reference_image[0] + r*np.cos(t), reference_image[1] + r*np.sin(t)], dtype=np.float32)
    geo_2 = flow.map_to_original_coord(geo.T)
    abs_geo_2 = copy.deepcopy(geo.T)

    # compute geodesics at range of angles:
    r = np.linspace(0.0, flow.sigma_to_length(4.), 1000)
    theta = np.linspace(0.0, 2.0*np.pi, 30)
    geodesics, abs_geodesics = [], []
    for t in theta:
        geo = np.array([reference_image[0] + r*np.cos(t), reference_image[1] + r*np.sin(t)], dtype=np.float32)
        geodesics.append(flow.map_to_original_coord(geo.T))
        abs_geodesics.append(geo.T)

    # plot geodesics
    plt.figure(figsize=(2*figsize[0], figsize[1]))
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    cmap = cm.get_cmap('Spectral')
    for ind, geo in enumerate(geodesics):
        ax1.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)
    ax1.plot(*np.array(geo_1).T, color='k', ls='--', zorder=-10, label='$\\theta=0$')
    ax1.plot(*np.array(geo_2).T, color='k', ls='-.', zorder=-10, label='$\\theta=\\pi/2$')

    for ind, geo in enumerate(abs_geodesics):
        ax2.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)
    ax2.plot(*np.array(abs_geo_1).T, color='k', ls='--', zorder=-10, label='$\\theta=0$')
    ax2.plot(*np.array(abs_geo_2).T, color='k', ls='-.', zorder=-10, label='$\\theta=\\pi/2$')

    # plot contours
    ax1.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    ax1.scatter(reference_point[0], reference_point[1], color='k')

    theta = np.linspace(0.0, 2.*np.pi, 200)
    ax2.plot(reference_image[0] + length*np.sin(theta), reference_image[1]+length*np.cos(theta), ls='--', lw=1., color='k', label='Contour centered at MAP')
    ax2.scatter(reference_image[0], reference_image[1], color='k', zorder=999)
    ax2.plot(length*np.sin(theta), length*np.cos(theta), ls='--', lw=1., color='k', alpha=.3, label='Contour centered at zero')

    ax1.set_xlim([np.amin(P1), np.amax(P1)])
    ax1.set_ylim([np.amin(P2), np.amax(P2)])
    ax1.set_xlabel(param_labels_latex[0], fontsize=fontsize)
    ax1.set_ylabel(param_labels_latex[1], fontsize=fontsize)
    ax2.set_xlabel('$Z_{1}$', fontsize=fontsize)
    ax2.set_ylabel('$Z_{2}$', fontsize=fontsize)

    ax1.legend()
    ax2.legend()
    plt.tight_layout()
    plt.savefig(outroot+'6_geodesics.pdf')
    plt.close('all')

    ###############################################################################
    # Plot of asymptotic structure
    ###############################################################################

    # feedback:
    print('7) asymptotic structure')

    # compute PCA of global covariance of samples:
    eig, eigv = np.linalg.eigh(cov_samples)

    # compute geodesics aligned with abstract coordinate axes:

    # computing a scale in order to observe asyntotic for parameters on different scales
    scale_x = abs(np.amax(P1) - np.amin(P1))
    scale_y = abs(np.amax(P2) - np.amin(P2))
    scale_r = np.linalg.norm([scale_x, scale_y])
    scale_r = 100

    r = np.linspace(-scale_r, scale_r, 1000)
    t = 0.0
    geo = np.array([reference_image[0] + r*np.cos(t), reference_image[1] + r*np.sin(t)], dtype=np.float32)
    geo_1 = flow.map_to_original_coord(geo.T)
    t = np.pi/2.
    geo = np.array([reference_image[0] + r*np.cos(t), reference_image[1] + r*np.sin(t)], dtype=np.float32)
    geo_2 = flow.map_to_original_coord(geo.T)

    # compute geodesics at range of angles:
    r = np.linspace(0.0, scale_r, 1000)
    theta = np.linspace(0.0, 2.0*np.pi, 1000)
    geodesics = []
    for t in theta:
        geo = np.array([reference_image[0] + r*np.cos(t),
                        reference_image[1] + r*np.sin(t)], dtype=np.float32)
        geodesics.append(flow.map_to_original_coord(geo.T))

    # plot geodesics:
    plt.figure(figsize=figsize)

    cmap = cm.get_cmap('Spectral')
    for ind, geo in enumerate(geodesics):
        plt.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)
    plt.plot(*np.array(geo_1).T, color='k', ls='--', zorder=-10, label='$\\theta=0$')
    plt.plot(*np.array(geo_2).T, color='k', ls='-.', zorder=-10, label='$\\theta=\\pi/2$')

    # plot PCA of covariance of samples
    for mode in [0, 1]:
        plt.axline(reference_point, reference_point+eig[mode]*eigv[:, mode], ls='-', color='k')

    # plot contours and MAP
    plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.scatter(reference_point[0], reference_point[1], color='k')

    plt.legend()
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'7_asymptotic_structure_of_geodesics.pdf')
    plt.close('all')

    ###########################################################################
    # Plot of local metric eigenvectors
    ###########################################################################

    # feedback:
    print('8) local metric eigenvectors')

    # restructure meshgrid of points to give an array of coordinates
    coords = np.array([coarse_X, coarse_Y], dtype=np.float32).reshape(2, -1).T

    # compute the metric at all coordinates
    local_metrics = flow.metric(coords)

    # compute the PCA eigenvalues and eigenvectors of each local metric
    PCA_eig, PCA_eigv = np.linalg.eigh(local_metrics)

    # sort PCA so first mode is index 0
    idx = np.argsort(PCA_eig, axis=1)[0]
    PCA_eig = PCA_eig[:, idx]
    PCA_eigv = PCA_eigv[:, :, idx]

    # plot PCA eigenvectors
    mode = 0
    plt.figure(figsize=figsize)
    plt.quiver(coords[:, 0], coords[:, 1], PCA_eigv[:, 0, mode], PCA_eigv[:, 1, mode], color='red', angles='xy', label='First mode')
    mode = 1
    plt.quiver(coords[:, 0], coords[:, 1], PCA_eigv[:, 0, mode], PCA_eigv[:, 1, mode], color='cadetblue', angles='xy', label='Second mode')

    # plot contours
    plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.scatter(reference_point[0], reference_point[1], color='k')

    # compute and plot eigenvectors of covariance of samples
    eig, eigv = np.linalg.eigh(cov_samples)
    for mode in [0, 1]:
        plt.axline(reference_point, reference_point+eig[mode]*eigv[:, mode], ls='-', color='k')

    plt.legend()
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'8_local_metric_PCA.pdf')
    plt.close('all')

    ###########################################################################
    # Plots of local metric eigenvalues
    ###########################################################################

    # feedback:
    print('9) local metric eigenvalues')

    coords = np.array([X, Y], dtype=np.float32).reshape(2, -1).T

    # compute the metric at all coordinates
    local_metrics = flow.metric(coords)

    # compute the PCA eigenvalues and eigenvectors of each local metric
    PCA_eig, PCA_eigv = np.linalg.eigh(local_metrics)

    # sort PCA so first mode is index 0
    idx = np.argsort(PCA_eig, axis=1)[0]
    PCA_eig = PCA_eig[:, idx]
    PCA_eigv = PCA_eigv[:, :, idx]

    # plot PCA eigenvalues
    plt.figure(figsize=(2.4*figsize[0], figsize[1]))
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    mode = 0
    pc = ax1.pcolormesh(X, Y, np.log10(PCA_eig[:, mode].reshape(200, 200)), linewidth=0, rasterized=True, shading='auto', cmap='BrBG_r', label='First mode')
    colorbar = plt.colorbar(pc, ax=ax1)
    # plot contours
    ax1.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    ax1.scatter(reference_point[0], reference_point[1], color='k')
    ax1.set_xlim([np.amin(P1), np.amax(P1)])
    ax1.set_ylim([np.amin(P2), np.amax(P2)])
    ax1.set_xlabel(param_labels_latex[0], fontsize=fontsize)
    ax1.set_ylabel(param_labels_latex[1], fontsize=fontsize)
    ax1.set_title('Mode 0')

    mode = 1
    pc = ax2.pcolormesh(X, Y, np.log10(PCA_eig[:, mode].reshape(200, 200)), linewidth=0, rasterized=True, shading='auto', cmap='BrBG_r', label='Second mode')
    colorbar = plt.colorbar(pc, ax=ax2)
    # plot contours
    ax2.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    ax2.scatter(reference_point[0], reference_point[1], color='k')
    ax2.set_xlim([np.amin(P1), np.amax(P1)])
    ax2.set_ylim([np.amin(P2), np.amax(P2)])
    ax2.set_xlabel(param_labels_latex[0], fontsize=fontsize)
    ax2.set_ylabel(param_labels_latex[1], fontsize=fontsize)
    ax2.set_title('Mode 1')

    plt.tight_layout()
    plt.savefig(outroot+'9_local_metric_PCA_eigs.pdf')
    plt.close('all')

    ###########################################################################
    # Plot eigenvalue network:
    ###########################################################################

    # feedback:
    print('10) PCA flow')

    def eigenvalue_ode(t, y, reference):
        """
        Solve the dynamical equation for eigenvalues.
        """
        # preprocess:
        x = tf.convert_to_tensor([tf.cast(y, tf.float32)])
        # map to original space to compute Jacobian (without inversion):
        x_par = flow.map_to_original_coord(x)
        # precompute Jacobian and its derivative:
        jac = flow.inverse_jacobian(x_par)[0]
        jac_T = tf.transpose(jac)
        jac_jac_T = tf.matmul(jac, jac_T)
        # compute eigenvalues:
        eig, eigv = tf.linalg.eigh(jac_jac_T)
        temp = tf.matmul(tf.transpose(eigv), tf.transpose([reference]))
        idx = tf.math.argmax(tf.abs(temp))[0]
        w = tf.convert_to_tensor([tf.math.sign(temp[idx]) * eigv[:, idx]])
        #
        return w

    def solve_eigenvalue_ode_abs(y0, n, length=1.5, num_points=100, **kwargs):
        """
        Solve eigenvalue problem in abstract space
        """
        # define solution points:
        solution_times = tf.linspace(0., length, num_points)
        # compute initial PCA:
        x_abs = tf.convert_to_tensor([y0])
        x_par = flow.map_to_original_coord(x_abs)
        jac = flow.inverse_jacobian(x_par)[0]
        jac_T = tf.transpose(jac)
        jac_jac_T = tf.matmul(jac, jac_T)
        # compute eigenvalues:
        eig, eigv = tf.linalg.eigh(jac_jac_T)
        # initialize solution:
        temp_sol_1 = np.zeros((num_points-1, flow.num_params))
        temp_sol_dot_1 = np.zeros((num_points-1, flow.num_params))
        temp_sol_2 = np.zeros((num_points-1, flow.num_params))
        temp_sol_dot_2 = np.zeros((num_points-1, flow.num_params))
        # integrate forward:
        solver = scipy.integrate.ode(eigenvalue_ode)
        #solver.set_integrator('lsoda')
        solver.set_initial_value(y0, 0.)
        reference = eigv[:, n]
        yt = y0.numpy()
        for ind, t in enumerate(solution_times[1:]):
            # set the reference:
            solver.set_f_params(reference)
            # advance solver:
            try:
                yt = solver.integrate(t)
            except:
                pass
            # compute derivative after time-step:
            yprime = eigenvalue_ode(t, yt, reference)
            # update reference:
            reference = yprime[0]
            # save out:
            temp_sol_1[ind] = yt.copy()
            temp_sol_dot_1[ind] = yprime.numpy().copy()
        # integrate backward:
        solver = scipy.integrate.ode(eigenvalue_ode)
        #solver.set_integrator()
        solver.set_initial_value(y0, 0.)
        reference = - eigv[:, n]
        yt = y0.numpy()
        for ind, t in enumerate(solution_times[1:]):
            # set the reference:
            solver.set_f_params(reference)
            # advance solver:
            try:
                yt = solver.integrate(t)
            except:
                pass
            # compute derivative after time-step:
            yprime = eigenvalue_ode(t, yt, reference)
            # update reference:
            reference = yprime[0]
            # save out:
            temp_sol_2[ind] = yt.copy()
            temp_sol_dot_2[ind] = yprime.numpy().copy()
        # patch solutions:
        times = np.concatenate((-solution_times[::-1], solution_times[1:]))
        traj = np.concatenate((temp_sol_2[::-1], x_abs.numpy(), temp_sol_1))
        vel = np.concatenate((-temp_sol_dot_2[::-1], [eigv[:, n].numpy()], temp_sol_dot_1))
        #
        return times, traj, vel

    def solve_eigenvalue_ode_par(y0, n, length=1.5, num_points=100, **kwargs):
        """
        Solve eigenvalue ODE in parameter space
        """
        # go to abstract space:
        x_par = tf.convert_to_tensor([y0])
        x_abs = flow.map_to_abstract_coord(x_par)[0]
        # call solver:
        times, traj, vel = solve_eigenvalue_ode_abs(x_abs, n, length=length, num_points=num_points, **kwargs)
        # convert back:
        traj = flow.map_to_original_coord(tf.cast(traj, tf.float32))
        #
        return times, traj

    # lines along the global principal components:
    y0 = flow.cast(reference_point)
    length = (flow.sigma_to_length(3)).astype(np.float32)

    _, temp_start_1 = solve_eigenvalue_ode_par(flow.cast(y0), n=0, length=length, num_points=101, side='+')
    _, temp_start_2 = solve_eigenvalue_ode_par(flow.cast(y0), n=0, length=length, num_points=101, side='-')
    start_1 = np.concatenate((temp_start_2[::101//5][:-1], temp_start_1[::101//5]))

    _, temp_start_1 = solve_eigenvalue_ode_par(flow.cast(y0), n=1, length=length, num_points=101, side='+')
    _, temp_start_2 = solve_eigenvalue_ode_par(flow.cast(y0), n=1, length=length, num_points=101, side='-')
    start_0 = np.concatenate((temp_start_2[::101//5][:-1], temp_start_1[::101//5]))

    # solve:
    modes_0, modes_1 = [], []
    for start in start_0:
        _, mode = solve_eigenvalue_ode_par(start, n=0, length=length, num_points=100)
        modes_0.append(mode)
    for start in start_1:
        _, mode = solve_eigenvalue_ode_par(start, n=1, length=length, num_points=100)
        modes_1.append(mode)

    # plot:
    plt.figure(figsize=(2*figsize[0], figsize[1]))
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    for mode in modes_0:
        ax1.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='k')
    for mode in modes_1:
        ax1.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='red')

    ax1.contour(X, Y, P, get_levels(P, x, y, levels_3), linewidths=2., linestyles='-', colors=['blue' for i in levels_5], zorder=999)
    ax1.scatter(reference_point[0], reference_point[1], color='k')

    ax1.set_xlim([np.amin(P1), np.amax(P1)])
    ax1.set_ylim([np.amin(P2), np.amax(P2)])
    ax1.set_xlabel(param_labels_latex[0], fontsize=fontsize)
    ax1.set_ylabel(param_labels_latex[1], fontsize=fontsize)

    for mode in modes_0:
        mode_abs = flow.map_to_abstract_coord(mode)
        ax2.plot(*np.array(mode_abs).T, lw=1., ls='-', color='k')
    for mode in modes_1:
        mode_abs = flow.map_to_abstract_coord(mode)
        ax2.plot(*np.array(mode_abs).T, lw=1., ls='-', color='red')

    # print the iso-contours:
    origin = [0, 0]
    theta = np.linspace(0.0, 2.*np.pi, 200)
    for i in range(4):
        _length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(i), 2))
        ax2.plot(origin[0]+_length*np.sin(theta), origin[1]+_length*np.cos(theta), ls='--', lw=2., color='blue')
    y0_abs = flow.map_to_abstract_coord(y0)
    ax2.scatter(y0_abs[0], y0_abs[1], color='k', zorder=999)

    ax2.set_xlabel('$Z_{1}$', fontsize=fontsize)
    ax2.set_ylabel('$Z_{2}$', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'11_local_pca_flow.pdf')

    ############################################################################
    # Find principal eigenvalue flow:
    ###########################################################################
    from scipy.interpolate import interp1d

    # feedback:
    print('12) PCA')

    def expected_distance(y0, max_length, n1, n2, samples):
        # solve the pca equation:
        time, mode = solve_eigenvalue_ode_par(y0, n=n1, length=max_length, num_points=100)
        # interpolate:
        interp_mode = interp1d(time, mode, kind='cubic', axis=0)
        # compute distances:
        distances = []
        for samp in samples:
            # solve:
            time2, mode2 = solve_eigenvalue_ode_par(samp, n=n2, length=max_length, num_points=100)
            # interpolate:
            interp_mode2 = interp1d(time2, mode2, kind='cubic', axis=0)
            # minimize distance:
            def _helper(y):
                lambda_1, lambda_2 = y
                return np.linalg.norm(interp_mode(lambda_1)-interp_mode2(lambda_2))
            result = scipy.optimize.minimize(_helper, [0., 0.],
                                             bounds=[[-max_length, max_length], [-max_length, max_length]])
            # append result:
            distances.append(result.x[1]**2)
        #
        return np.mean(distances)

    try:
        # general setup:
        y0 = flow.cast(reference_point)
        length = (flow.sigma_to_length(3)).astype(np.float32)

        # minimize for first mode:
        time_0, start_0 = solve_eigenvalue_ode_par(y0, n=1, length=length, num_points=100)
        interp_start = interp1d(time_0, start_0, kind='cubic', axis=0)
        num_samples = 100
        samples = flow.sample(num_samples)
        def _helper_temp(temp):
            return expected_distance(interp_start(temp[0]).astype(np.float32), length, n1=0, n2=1, samples=samples)
        if use_MAP:
            try:
                result_0 = scipy.optimize.minimize(_helper_temp, [0.],
                                                   bounds=[[-length, length]], method='L-BFGS-B')
                result_0 = result_0.x
            except:
                result_0 = 0.0
        else:
            result_0 = 0.0
        pca_mode_0_times, pca_mode_0 = solve_eigenvalue_ode_par(interp_start(result_0)[0].astype(np.float32), n=0, length=length, num_points=100)

        # minimize for the second mode:
        time_1, start_1 = solve_eigenvalue_ode_par(y0, n=0, length=length, num_points=100)
        interp_start = interp1d(time_1, start_1, kind='cubic', axis=0)
        num_samples = 100
        samples = flow.sample(num_samples)
        def _helper_temp(temp):
            return expected_distance(interp_start(temp[0]).astype(np.float32), length, n1=1, n2=0, samples=samples)
        if use_MAP:
            result_1 = scipy.optimize.minimize(_helper_temp, [0.0],
                                               bounds=[[-length, length]], method='L-BFGS-B')
            result_1 = result_1.x
        else:
            result_1 = 0.0
        pca_mode_1_times, pca_mode_1 = solve_eigenvalue_ode_par(interp_start(result_1)[0].astype(np.float32), n=1, length=length, num_points=100)

        # plot in parameter space:
        plt.figure(figsize=(2*figsize[0], figsize[1]))
        gs = gridspec.GridSpec(1, 2)
        ax1 = plt.subplot(gs[0, 0])
        ax2 = plt.subplot(gs[0, 1])

        for mode in modes_0:
            ax1.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='k')
        ax1.plot(pca_mode_0[:, 0], pca_mode_0[:, 1], lw=2., ls='-', color='k')
        for mode in modes_1:
            ax1.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='red')
        ax1.plot(pca_mode_1[:, 0], pca_mode_1[:, 1], lw=2., ls='-', color='red')
        ax1.contour(X, Y, P, get_levels(P, x, y, levels_3), linewidths=2., linestyles='-', colors=['blue' for i in levels_5], zorder=999)
        ax1.scatter(maximum_posterior[0], maximum_posterior[1], color='k')
        ax1.set_xlim([np.amin(P1), np.amax(P1)])
        ax1.set_ylim([np.amin(P2), np.amax(P2)])
        ax1.set_xlabel(param_labels_latex[0], fontsize=fontsize)
        ax1.set_ylabel(param_labels_latex[1], fontsize=fontsize)

        # plot in abstract space:
        for mode in modes_0:
            mode_abs = flow.map_to_abstract_coord(mode)
            ax2.plot(*np.array(mode_abs).T, lw=1., ls='-', color='k')
        mode_abs = flow.map_to_abstract_coord(pca_mode_0)
        ax2.plot(mode_abs[:, 0], mode_abs[:, 1], lw=2., ls='-', color='k')
        for mode in modes_1:
            mode_abs = flow.map_to_abstract_coord(mode)
            ax2.plot(*np.array(mode_abs).T, lw=1., ls='-', color='red')
        mode_abs = flow.map_to_abstract_coord(pca_mode_1)
        ax2.plot(mode_abs[:, 0], mode_abs[:, 1], lw=2., ls='-', color='red')

        # print the iso-contours:
        origin = [0, 0]  # flow.map_to_abstract_coord(y0)
        theta = np.linspace(0.0, 2.*np.pi, 200)
        for i in range(4):
            _length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(i), 2))
            ax2.plot(origin[0]+_length*np.sin(theta), origin[1]+_length*np.cos(theta), ls='--', lw=2., color='blue')
        y0_abs = flow.map_to_abstract_coord(y0)
        ax2.scatter(y0_abs[0], y0_abs[1], color='k', zorder=999)

        ax2.set_xlabel('$Z_{1}$', fontsize=fontsize)
        ax2.set_ylabel('$Z_{2}$', fontsize=fontsize)
        plt.tight_layout()
        plt.savefig(outroot+'12_local_pca_abstract.pdf')
        plt.close('all')

        ###########################################################################
        # Plot probability along principal eigenvalue flow:
        ###########################################################################

        plt.figure(figsize=figsize)

        points = pca_mode_0
        probs = flow.log_probability(points)
        points_abs = flow.map_to_abstract_coord(points)
        v = points_abs - maximum_posterior_abs
        dists = np.linalg.norm(v, axis=1)
        dists[:np.argmin(dists)] *= -1
        plt.plot(dists, probs, label='Mode 0', c='k')

        points = pca_mode_1
        probs = flow.log_probability(points)
        points_abs = flow.map_to_abstract_coord(points)
        v = points_abs - maximum_posterior_abs
        dists = np.linalg.norm(v, axis=1)
        dists[:np.argmin(dists)] *= -1
        plt.plot(dists, probs, label='Mode 1', c='r')

        plt.xlabel('Distance from Maximum Posterior along principal mode', fontsize=fontsize)
        plt.ylabel('Log Probability', fontsize=fontsize)
        plt.legend()
        plt.tight_layout()
        plt.savefig(outroot+'13_probability_local_pca.pdf')
        plt.close('all')

    except Exception as ex:
        print(ex)
