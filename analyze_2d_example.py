# -*- coding: utf-8 -*-

"""
General code to analyze examples

For testing purposes:

import example_3_generate as example

posterior_chain = example.posterior_chain
prior_chain = example.prior_chain
param_names = posterior_chain.getParamNames().list()
outroot = example.out_folder
train_params = {}
param_ranges = None
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
        res = optimize.brentq(lambda x: _helper(x)-c, np.amin(P), np.amax(P))
        levs.append(res)
    return levs


###############################################################################

def run_example_2d(posterior_chain, prior_chain, param_names, outroot, param_ranges=None, train_params={}):
    """
    Run full analysis of 2d example case, as in flow playground
    """

    # plotting preferences:
    figsize = (8, 8)
    fontsize = 15

    # parameter ranges for plotting from the prior:
    if param_ranges is None:
        param_ranges = np.array([np.amin(prior_chain.samples, axis=0), np.amax(prior_chain.samples, axis=0)]).T
        param_ranges = param_ranges[[prior_chain.index[name] for name in param_names], :]

    # parameter labels:
    param_labels = [name.label for name in posterior_chain.getParamNames().parsWithNames(param_names)]
    param_labels_latex = ['$'+name+'$' for name in param_labels]

    # obtain the synthetic probability:
    flow_P = synthetic_probability.DiffFlowCallback(posterior_chain, param_names=param_names, feedback=1, learning_rate=0.01)
    batch_size = train_params.get('batch_size', 8192)
    epochs = train_params.get('epochs', 40)
    steps_per_epoch = train_params.get('steps_per_epoch', 128)
    flow_P.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

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
    density = posterior_chain.get2DDensity(param_names[0], param_names[1], normalized=True)
    _X, _Y = np.meshgrid(density.x, density.y)
    density.P = density.P / simps(simps(density.P, density.y), density.x)

    # levels for contour plots:
    levels_5 = [utilities.from_sigma_to_confidence(i) for i in range(5, 1, -1)]
    levels_3 = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]

    ###########################################################################
    # Plot learned distribution from samples:
    ###########################################################################

    N = 10000
    X_sample = np.array(flow_P.sample(N))
    flow_chain = MCSamples(samples=X_sample,
                           loglikes = -flow_P.log_probability(X_sample).numpy(),
                           names=param_names,
                           label='Learned distribution')

    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain, flow_chain], params=param_names, filled=False)
    g.export(outroot+'1_learned_posterior_distribution.pdf')

    ###########################################################################
    # Plot learned distribution using flow log probability (and samples):
    ###########################################################################

    # compute flow probability on a grid:
    log_P = flow_P.log_probability(np.array([X, Y], dtype=np.float32).T)
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

    ###########################################################################
    # Plot samples:
    ###########################################################################

    # in parameter space:
    plt.figure(figsize=figsize)
    plt.scatter(flow_chain.samples[:, 0], flow_chain.samples[:, 1], s=0.3, c=flow_chain.loglikes)
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'2_learned_distribution_samples.pdf')

    # in abstract space:
    abstract_samples = flow_P.map_to_abstract_coord(flow_chain.samples)
    plt.figure(figsize=figsize)
    plt.scatter(abstract_samples[:, 0], abstract_samples[:, 1], s=0.3, c=flow_chain.loglikes)
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'2_learned_distribution_samples_abstract.pdf')

    ###########################################################################
    # Plot log determinant of metric:
    ###########################################################################

    # compute log determinant of metric:
    log_det = flow_P.log_det_metric(np.array([X, Y], dtype=np.float32).T)
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

    ###########################################################################
    # Plot maximum posterior and mean:
    ###########################################################################

    # find the MAP:
    result = flow_P.MAP_finder(disp=True)
    maximum_posterior = result.x
    # mean:
    mean = posterior_chain.getMeans([posterior_chain.index[name] for name in param_names])

    # plot contours with MAP and mean
    plt.figure(figsize=figsize)
    plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='green', label='MAP: (%.3f, %.3f)' %(maximum_posterior[0],maximum_posterior[1]))
    plt.scatter(mean[0], mean[1], color='red', label='mean: (%.3f, %.3f)' %(mean[0],mean[1]))
    plt.legend()
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'4_maximum_posterior_and_sample_mean.pdf')

    ###########################################################################
    # Plot comparison of covariance metric, fisher metric,
    ###########################################################################

    # covariance from samples
    cov_samples = posterior_chain.cov(pars=param_names)

    # metrics from flow around mean:
    covariance_metric = flow_P.metric(np.array([mean]).astype(np.float32))[0]
    fisher_metric = flow_P.inverse_metric(np.array([mean]).astype(np.float32))[0]

    alpha = np.linspace(-1, 1, 1000)
    plt.figure(figsize=figsize)

    # plot PCA of flow covariance metric
    _, eigv = np.linalg.eigh(covariance_metric)
    mode = 0
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='k', ls='--', label='flow covariance')
    mode = 1
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='k', ls='--')

    # plot PCA of flow fisher metric
    _, eigv = np.linalg.eigh(fisher_metric)
    mode = 0
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='green', ls='-.', label='flow fisher')
    mode = 1
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='green', ls='-.')

    # plot PCA of covariance of samples
    _, eigv = np.linalg.eigh(cov_samples)
    mode = 0
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='red', ls='-', label='samples')
    mode = 1
    plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='red', ls='-')

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

    ###########################################################################
    # trace geodesics in abstract space passing from the maximum posterior
    ###########################################################################

    # find where the MAP goes:
    map_image = flow_P.map_to_abstract_coord(np.array(maximum_posterior, dtype=np.float32))

    # compute geodesics aligned with abstract coordinate axes:
    r = np.linspace(-20.0, 20.0, 1000)
    t = 0.0
    geo = np.array([map_image[0] + r*np.cos(t), map_image[1] + r*np.sin(t)], dtype=np.float32)
    geo_1 = flow_P.map_to_original_coord(geo.T)
    t = np.pi/2.
    geo = np.array([map_image[0] + r*np.cos(t), map_image[1] + r*np.sin(t)], dtype=np.float32)
    geo_2 = flow_P.map_to_original_coord(geo.T)

    # compute geodesics at range of angles:
    r = np.linspace(0.0, 20.0, 1000)
    theta = np.linspace(0.0, 2.0*np.pi, 30)
    geodesics = []
    for t in theta:
        geo = np.array([map_image[0] + r*np.cos(t), map_image[1] + r*np.sin(t)], dtype=np.float32)
        geodesics.append(flow_P.map_to_original_coord(geo.T))

    # plot geodesics
    plt.figure(figsize=figsize)
    cmap = cm.get_cmap('Spectral')
    for ind, geo in enumerate(geodesics):
        plt.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)
    plt.plot(*np.array(geo_1).T, color='k', ls='--', zorder=-10, label='$\\theta=0$')
    plt.plot(*np.array(geo_2).T, color='k', ls='-.', zorder=-10, label='$\\theta=\\pi/2$')

    # plot contours
    plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.legend()
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'6_abstract_geodesics_in_parameter_space.pdf')

    ###############################################################################
    # Plot of asymptotic structure
    ###############################################################################

    # compute PCA of global covariance of samples:
    eig, eigv = np.linalg.eigh(cov_samples)

    # compute geodesics aligned with abstract coordinate axes:

    ## computing a scale in order to observe asyntotic for parameters on different scales
    scale_x = abs(np.amax(P1) - np.amin(P1))
    scale_y = abs(np.amax(P2) - np.amin(P2))
    scale_r = np.linalg.norm([scale_x, scale_y])

    r = np.linspace(-100000*scale_r, 100000.0*scale_r, 100)
    t = 0.0
    geo = np.array([map_image[0] + r*np.cos(t), map_image[1] + r*np.sin(t)], dtype=np.float32)
    geo_1 = flow_P.map_to_original_coord(geo.T)
    t = np.pi/2.
    geo = np.array([map_image[0] + r*np.cos(t), map_image[1] + r*np.sin(t)], dtype=np.float32)
    geo_2 = flow_P.map_to_original_coord(geo.T)

    # compute geodesics at range of angles:
    r = np.linspace(0.0, 100000.0*scale_r, 100)
    theta = np.linspace(0.0, 2.0*np.pi, 100)
    geodesics = []
    for t in theta:
        geo = np.array([map_image[0] + r*np.cos(t),
                        map_image[1] + r*np.sin(t)], dtype=np.float32)
        geodesics.append(flow_P.map_to_original_coord(geo.T))

    # plot geodesics:
    plt.figure(figsize=figsize)

    cmap = cm.get_cmap('Spectral')
    for ind, geo in enumerate(geodesics):
        plt.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)
    plt.plot(*np.array(geo_1).T, color='k', ls='--', zorder=-10, label='$\\theta=0$')
    plt.plot(*np.array(geo_2).T, color='k', ls='-.', zorder=-10, label='$\\theta=\\pi/2$')

    # plot PCA of covariance of samples
    r = np.linspace(-100000*scale_r, 100000.0*scale_r, 100)
    mode = 0
    plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')
    mode = 1
    plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')

    # plot contours and MAP
    plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

    plt.xlim([-1000*scale_x, 1000.0*scale_x])
    plt.ylim([-1000*scale_y, 1000.0*scale_y])
    plt.legend()
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'7_asymptotic_structure_of_geodesics.pdf')

    ###########################################################################
    # Plot of local metric eigenvalues
    ###########################################################################

    # restructure meshgrid of points to give an array of coordinates
    coords = np.array([coarse_X, coarse_Y], dtype=np.float32).reshape(2, -1).T

    # compute the metric at all coordinates
    local_metrics = flow_P.metric(coords)

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
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

    # compute and plot eigenvalues of covariance of samples
    eig, eigv = np.linalg.eigh(cov_samples)
    mode = 0
    plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')
    mode = 1
    plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')

    plt.legend()
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'8_local_metric_PCA.pdf')

    ###########################################################################
    # Plot geodesics around MAP:
    ###########################################################################

    # define function to rotate vector by set angle theta:
    def rot(v, theta):
        rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        v_new = np.dot(rot, v)
        return v_new

    # initial point as MAP and determine eigenvector (not normalized):
    y_init = maximum_posterior.astype(np.float32)
    covariance_metric = flow_P.metric(np.array([y_init]).astype(np.float32))[0]
    eig, eigv = np.linalg.eigh(covariance_metric)
    yprime_init = eigv[:, 0]

    # define length to travel along geodesic using chi2
    length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(3), 2))
    solution_times = np.linspace(0., length, 200)

    # loop through angles and plot:
    geo_list = []
    plt.figure(figsize=figsize)
    theta_arr = np.linspace(0.0, 2.0*np.pi, 30)
    for ind, theta in enumerate(theta_arr):
        yprime = rot(yprime_init, theta).astype(np.float32)
        # normalize vector using metric
        norm = np.sqrt(np.dot(np.dot(yprime,covariance_metric),yprime))
        yprime /= norm
        results = flow_P.solve_geodesic(y_init, yprime,solution_times)
        geo = results.states[:, 0:2]
        geo_list.append(geo)
        #plt.quiver(results.states[:,0], results.states[:,1], results.states[:, 2], results.states[:, 3], color=cmap(ind/len(theta_arr)), angles = 'xy')
        plt.plot(results.states[:, 0], results.states[:, 1], ls='--', color=cmap(ind/len(theta_arr)))
    plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.savefig(outroot+'9_geodesics_around_MAP.pdf')

    # plot geodesics in abstract space:
    plt.figure(figsize=figsize)
    for ind, geo in enumerate(geo_list):
        geo = np.array(geo)
        geo_abs = flow_P.map_to_abstract_coord(geo)
        plt.plot(*np.array(geo_abs).T, ls='--', color=cmap(ind/len(geo_list)))

    # print the iso-contours:
    origin = flow_P.map_to_abstract_coord(y_init)
    theta = np.linspace(0.0, 2.*np.pi, 200)
    plt.plot(origin[0]+length*np.sin(theta), origin[1]+length*np.cos(theta), ls='--', lw=1., color='k')
    plt.scatter(origin[0], origin[1], color='k', zorder=999)

    plt.xlabel('$Z_{1}$', fontsize=fontsize)
    plt.ylabel('$Z_{2}$', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'10_geodesics_in_abstract_space.pdf')

    ###########################################################################
    # Plot eigenvalue network:
    ###########################################################################

    @tf.function()
    def eigenvalue_ode_temp(t, y, n, side=1.):
        # compute metric:
        metric = flow_P.metric(tf.convert_to_tensor([y]))
        # compute eigenvalues:
        eig, eigv = tf.linalg.eigh(metric[0])
        #
        return side * eigv[:, n] / tf.sqrt(eig[n])

    @tf.function()
    def solve_eigenvalue_ode_temp(y0, solution_times, n, side=1., **kwargs):
        # solve with explicit solver:
        results = tfp.math.ode.DormandPrince(rtol=1.e-4).solve(eigenvalue_ode_temp, initial_time=0., initial_state=y0, solution_times=solution_times, constants={'n': n, 'side': side})
        #
        return results

    def helper_solve_geo(y0, n, length=1.5, num_points=100):
        solution_times = np.linspace(0., length, num_points)
        temp_sol_1 = solve_eigenvalue_ode_temp(y0, solution_times, n=n, side=1.)
        temp_sol_2 = solve_eigenvalue_ode_temp(y0, solution_times, n=n, side=-1.)
        times = tf.concat([-temp_sol_2.times[1:][::-1], temp_sol_1.times], axis=0)
        traj = tf.concat([temp_sol_2.states[1:][::-1], temp_sol_1.states], axis=0)
        #
        return times, traj

    # obtain PCA modes that pass through MAP:
    y0 = maximum_posterior.astype(np.float32)
    length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(3), 2))
    _, start_1 = helper_solve_geo(y0, n=0, length=2.*length, num_points=10)
    _, start_0 = helper_solve_geo(y0, n=1, length=2.*length, num_points=5)

    modes_0, modes_1 = [], []
    plt.figure(figsize=figsize)
    for start in start_0:
        _, mode = helper_solve_geo(start, n=0, length=2.*length, num_points=100)
        modes_0.append(mode)
        plt.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='k')
    for start in start_1:
        _, mode = helper_solve_geo(start, n=1, length=2.*length, num_points=100)
        modes_1.append(mode)
        plt.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='red')
    plt.contour(X, Y, P, get_levels(P, x, y, levels_3), linewidths=2., linestyles='-', colors=['blue' for i in levels_5], zorder=999)
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'11_local_pca_flow.pdf')

    # plot in abstract space:
    plt.figure(figsize=figsize)
    for mode in modes_0:
        mode_abs = flow_P.map_to_abstract_coord(mode)
        plt.plot(*np.array(mode_abs).T, lw=1., ls='-', color='k')
    for mode in modes_1:
        mode_abs = flow_P.map_to_abstract_coord(mode)
        plt.plot(*np.array(mode_abs).T, lw=1., ls='-', color='red')

    # print the iso-contours:
    origin = flow_P.map_to_abstract_coord(y0)
    theta = np.linspace(0.0, 2.*np.pi, 200)
    for i in range(4):
        _length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(i), 2))
        plt.plot(origin[0]+_length*np.sin(theta), origin[1]+_length*np.cos(theta), ls='--', lw=2., color='blue')
    plt.scatter(origin[0], origin[1], color='k', zorder=999)

    plt.xlabel('$Z_{1}$', fontsize=fontsize)
    plt.ylabel('$Z_{2}$', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'12_local_pca_flow_abstract.pdf')

    ###########################################################################
    # Find principal eigenvalue flow:
    ###########################################################################
    from scipy.interpolate import interp1d

    def expected_distance(y0, max_length, n1, n2, samples):
        # solve the pca equation:
        time, mode = helper_solve_geo(y0, n=n1, length=max_length, num_points=100)
        # interpolate:
        interp_mode = interp1d(time, mode, kind='cubic', axis=0)
        # compute distances:
        distances = []
        for samp in samples:
            # solve:
            time2, mode2 = helper_solve_geo(samp, n=n2, length=max_length, num_points=100)
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

    # general setup:
    y0 = maximum_posterior.astype(np.float32)
    length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(3), 2))
    # minimize for first mode:
    time_0, start_0 = helper_solve_geo(y0, n=1, length=2.*length, num_points=100)
    interp_start = interp1d(time_0, start_0, kind='cubic', axis=0)
    num_samples = 100
    samples = flow_P.sample(num_samples)
    def _helper_temp(temp):
        return expected_distance(interp_start(temp[0]).astype(np.float32), 2.*length, n1=0, n2=1, samples=samples)
    result_0 = scipy.optimize.minimize(_helper_temp, [0.],
                                     bounds=[[-length, length]], method='L-BFGS-B')
    pca_mode_0_times, pca_mode_0 = helper_solve_geo(interp_start(result_0.x)[0].astype(np.float32), n=0, length=2.*length, num_points=100)

    # minimize for the second mode:
    time_1, start_1 = helper_solve_geo(y0, n=0, length=2.*length, num_points=100)
    interp_start = interp1d(time_1, start_1, kind='cubic', axis=0)
    num_samples = 100
    samples = flow_P.sample(num_samples)
    def _helper_temp(temp):
        return expected_distance(interp_start(temp[0]).astype(np.float32), 2.*length, n1=1, n2=0, samples=samples)
    result_1 = scipy.optimize.minimize(_helper_temp, [0.],
                                     bounds=[[-length, length]], method='L-BFGS-B')
    pca_mode_1_times, pca_mode_1 = helper_solve_geo(interp_start(result_1.x)[0].astype(np.float32), n=1, length=2.*length, num_points=100)

    # plot in parameter space:
    plt.figure(figsize=figsize)
    for mode in modes_0:
        plt.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='k')
    plt.plot(pca_mode_0[:, 0], pca_mode_0[:, 1], lw=2., ls='-', color='k')
    for mode in modes_1:
        plt.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='red')
    plt.plot(pca_mode_1[:, 0], pca_mode_1[:, 1], lw=2., ls='-', color='red')
    plt.contour(X, Y, P, get_levels(P, x, y, levels_3), linewidths=2., linestyles='-', colors=['blue' for i in levels_5], zorder=999)
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'13_local_pca.pdf')

    # plot in abstract space:
    plt.figure(figsize=figsize)
    for mode in modes_0:
        mode_abs = flow_P.map_to_abstract_coord(mode)
        plt.plot(*np.array(mode_abs).T, lw=1., ls='-', color='k')
    mode_abs = flow_P.map_to_abstract_coord(pca_mode_0)
    plt.plot(mode_abs[:, 0], mode_abs[:, 1], lw=2., ls='-', color='k')
    for mode in modes_1:
        mode_abs = flow_P.map_to_abstract_coord(mode)
        plt.plot(*np.array(mode_abs).T, lw=1., ls='-', color='red')
    mode_abs = flow_P.map_to_abstract_coord(pca_mode_1)
    plt.plot(mode_abs[:, 0], mode_abs[:, 1], lw=2., ls='-', color='red')

    # print the iso-contours:
    origin = flow_P.map_to_abstract_coord(y0)
    theta = np.linspace(0.0, 2.*np.pi, 200)
    for i in range(4):
        _length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(i), 2))
        plt.plot(origin[0]+_length*np.sin(theta), origin[1]+_length*np.cos(theta), ls='--', lw=2., color='blue')
    plt.scatter(origin[0], origin[1], color='k', zorder=999)

    plt.xlabel('$Z_{1}$', fontsize=fontsize)
    plt.ylabel('$Z_{2}$', fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(outroot+'14_local_pca_abstract.pdf')

    ###########################################################################
    # Run AI Feynman
    ###########################################################################

    ## save to file the first PCA mode:
    #with open('temp.txt', "w") as f:
    #    np.savetxt(f, pca_mode_0.numpy())
    #import aifeynman
    #aifeynman.run_aifeynman("./", "temp.txt", 60, "19ops.txt", polyfit_deg=2, NN_epochs=500)
    #plt.figure(figsize=figsize)
    #plt.plot(pca_mode_0[:, 0], pca_mode_0[:, 1], lw=1., ls='-', color='k')
    #plt.plot(pca_mode_1[:, 0], pca_mode_1[:, 1], lw=1., ls='-', color='red')
    #plt.contour(X, Y, P, get_levels(P, x, y, levels_3), linewidths=1., linestyles='-', colors=['blue' for i in levels_5], zorder=999)
    #plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')
    #plt.xlim([np.amin(P1), np.amax(P1)])
    #plt.ylim([np.amin(P2), np.amax(P2)])
    #plt.xlabel(param_labels_latex[0], fontsize=fontsize)
    #plt.ylabel(param_labels_latex[1], fontsize=fontsize)
    #plt.tight_layout()
