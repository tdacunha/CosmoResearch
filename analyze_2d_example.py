# -*- coding: utf-8 -*-

"""
General code to analyze examples

For testing purposes:

import example_2_generate as example

posterior_chain = example.posterior_chain
prior_chain = example.prior_chain
param_names = posterior_chain.getParamNames().list()
outroot = example.out_folder
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

# import the tensiometer tools that we need:
from tensiometer import utilities
from tensiometer import gaussian_tension
from tensiometer import mcmc_tension



###############################################################################

def run_example_2d(posterior_chain, prior_chain, param_names, param_ranges, outroot):
    """
    Run full analysis of 2d example case, as in flow playground
    """
    # plotting preferences:
    figsize = (8,8)
    fontsize = 15

    # obtain the synthetic probability:
    flow_P = synthetic_probability.DiffFlowCallback(posterior_chain, param_names=param_names, feedback=1, learning_rate=0.01)
    batch_size = 8192
    epochs = 40
    steps_per_epoch = 128
    flow_P.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    # obtain the parameter ranges:
    P1 = np.linspace(param_ranges[0][0],param_ranges[0][1], 20)
    P2 = np.linspace(param_ranges[1][0],param_ranges[1][1], 20)
    x, y = P1, P2
    X, Y = np.meshgrid(x, y)

    # obtain sample densities
    density = posterior_chain.get2DDensity(param_names[0], param_names[1], normalized=True)
    _X, _Y = np.meshgrid(density.x, density.y)

    # levels for contour plots:
    levels_5 = [utilities.from_sigma_to_confidence(i) for i in range(5, 1, -1)]
    levels_3 = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]


    ###############################################################################
    # PLOT LEARNED DISTRIBUTION FROM SAMPLES:
    ###############################################################################

    N = 10000
    X_sample = np.array(flow_P.sample(N))
    flow_chain = MCSamples(samples=X_sample, names=param_names, label='Learned distribution')

    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain, flow_chain], params=param_names, filled=False)
    g.export(outroot+'1_learned_posterior_distribution.pdf')

    ###############################################################################
    # PLOT LEARNED DISTRIBUTION USING FLOW LOG PROBABILITY (AND SAMPLES):
    ###############################################################################
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


    # using the method implemented in dist_learned:
    P1_cont = np.linspace(1.2*param_ranges[0][0],1.2*param_ranges[0][1], 200)
    P2_cont = np.linspace(1.2*param_ranges[1][0],1.2*param_ranges[1][1], 200)

    x_cont, y_cont = P1_cont, P2_cont
    X_cont, Y_cont = np.meshgrid(x_cont, y_cont)
    log_P = flow_P.log_probability(np.array([X_cont, Y_cont], dtype=np.float32).T)
    log_P = np.array(log_P).T
    P = np.exp(log_P)
    P = P / simps(simps(P, y_cont), x_cont)

    # plot learned contours
    plt.figure(figsize = figsize)
    plt.contour(X_cont, Y_cont, P, get_levels(P, x_cont, y_cont, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
    plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels_5), linewidths=1., linestyles='--', colors=['red' for i in levels_5])
    plt.xlabel(param_names[0], fontsize = fontsize)
    plt.ylabel(param_names[1], fontsize = fontsize)
    plt.savefig(outroot+'2_log_prob_distribution.pdf')
    plt.show()
    ###############################################################################
    # PLOT LOG DETERMINANT OF METRIC
    ###############################################################################

    # compute log determinant of metric:
    log_det = flow_P.log_det_metric(np.array([X, Y], dtype=np.float32).T)
    log_det = np.array(log_det).T

    # plot meshgrid of log determinant
    plt.figure(figsize = figsize)
    pc = plt.pcolormesh(X, Y, log_det, linewidth=0, rasterized=True, shading='auto', cmap='RdBu')
    colorbar = plt.colorbar(pc)

    # plot contours
    plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels_5), linewidths=1., linestyles='--', colors=['red' for i in levels_5])
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_names[0], fontsize = fontsize)
    plt.ylabel(param_names[1], fontsize = fontsize)
    plt.savefig(outroot+'3_log_det_jacobian_distribution.pdf')
    plt.show()

    ###############################################################################
    # PLOT MAXIMUM POSTERIOR AND MEAN
    ###############################################################################
    # MAP:
    result = flow_P.MAP_finder(disp=True)
    maximum_posterior = result.x
    # mean:
    mean = posterior_chain.getMeans([posterior_chain.index[name] for name in param_names])

    # plot contours with MAP and mean
    plt.figure(figsize = figsize)
    plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels_3), linewidths=1., linestyles='--', colors=['red' for i in levels_3])
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='green', label='MAP: (%.3f, %.3f)' %(maximum_posterior[0],maximum_posterior[1]))
    plt.scatter(mean[0], mean[1], color='red', label='mean: (%.3f, %.3f)' %(mean[0],mean[1]))
    plt.legend()
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.xlabel(param_names[0], fontsize = fontsize)
    plt.ylabel(param_names[1], fontsize = fontsize)
    plt.savefig(outroot+'4_maximum_posterior_and_sample_mean.pdf')
    plt.show()

    ###############################################################################
    # PLOT COMPARISON OF COVARIANCE METRIC, FISHER METRIC,
    ###############################################################################
    # covariance from samples
    cov_samples = posterior_chain.cov(pars=param_names)
    #fisher_samples = np.linalg.inv(cov_samples)

    # metrics from flow around mean:
    covariance_metric = flow_P.metric(np.array([mean]).astype(np.float32))[0]
    fisher_metric = flow_P.inverse_metric(np.array([mean]).astype(np.float32))[0]

    alpha = np.linspace(-1, 1, 1000)
    plt.figure(figsize = figsize)

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
    plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels_3), linewidths=1., linestyles='--', colors=['red' for i in levels_3])
    plt.scatter(mean[0], mean[1], color='k')

    plt.legend()
    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.legend()
    plt.xlabel(param_names[0], fontsize = fontsize)
    plt.ylabel(param_names[1], fontsize = fontsize)
    plt.savefig(outroot+'5_comparison_of_cov_fisher_samples_at_mean.pdf')
    plt.show()
    ###############################################################################
    # trace geodesics in abstract space passing from the maximum posterior
    ###############################################################################
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
    plt.figure(figsize = figsize)
    cmap = cm.get_cmap('Spectral')
    for ind, geo in enumerate(geodesics):
        plt.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)
    plt.plot(*np.array(geo_1).T, color='k', ls='--', zorder=-10, label='$\\theta=0$')
    plt.plot(*np.array(geo_2).T, color='k', ls='-.', zorder=-10, label='$\\theta=\\pi/2$')

    # plot contours
    plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels_3), linewidths=1., linestyles='-', colors=['k' for i in levels_3], zorder=0)
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

    plt.xlim([np.amin(P1), np.amax(P1)])
    plt.ylim([np.amin(P2), np.amax(P2)])
    plt.legend()
    plt.xlabel(param_names[0], fontsize = fontsize)
    plt.ylabel(param_names[1], fontsize = fontsize)
    plt.savefig(outroot+'6_abstract_geodesics_in_parameter_space.pdf')
    plt.show()

    ###############################################################################
    # PLOT OF ASYMPTOTIC STRUCTURE
    ###############################################################################
    # compute PCA of global covariance of samples:
    eig, eigv = np.linalg.eigh(cov_samples)

    # compute geodesics aligned with abstract coordinate axes:

    ## computing a scale in order to observe asyntotic for parameters on different scales
    scale_x = abs(np.amax(P1) - np.amin(P1))
    scale_y = abs(np.amax(P2) - np.amin(P2))
    scale_r = np.linalg.norm([scale_x,scale_y])

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
    plt.figure(figsize = figsize)

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
    plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels_3), linewidths=1., linestyles='-', colors=['k' for i in levels_3], zorder=0)
    plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

    plt.xlim([-1000*scale_x, 1000.0*scale_x])
    plt.ylim([-1000*scale_y, 1000.0*scale_y])
    plt.legend()
    plt.xlabel(param_names[0], fontsize = fontsize)
    plt.ylabel(param_names[1], fontsize = fontsize)
    plt.savefig(outroot+'7_asymptotic_structure_of_geodesics.pdf')
    plt.show()

    ###############################################################################
    # PLOT OF LOCAL METRIC EIGENVALUES
    ###############################################################################
    # restructure meshgrid of points to give an array of coordinates
    coords = np.array([X, Y], dtype = np.float32).reshape(2,-1).T

    # compute the metric at all coordinates
    local_metrics = flow_P.metric(coords)

    # compute the PCA eigenvalues and eigenvectors of each local metric
    PCA_eig, PCA_eigv = np.linalg.eigh(local_metrics)

    # sort PCA so first mode is index 0
    idx = np.argsort(PCA_eig, axis = 1)[0]
    PCA_eig = PCA_eig[:,idx]
    PCA_eigv = PCA_eigv[:,:,idx]

    # plot PCA eigenvectors
    mode = 0
    plt.figure(figsize = figsize)
    plt.quiver(coords[:,0], coords[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:, 1,mode], color = 'red', angles = 'xy', label = 'First mode')
    mode = 1
    plt.quiver(coords[:,0], coords[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:,1, mode], color = 'cadetblue', angles = 'xy', label = 'Second mode')

    # plot contours
    plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels_3), linewidths=1., linestyles='-', colors=['k' for i in levels_3], zorder=0)
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
    plt.xlabel(param_names[0], fontsize = fontsize)
    plt.ylabel(param_names[1], fontsize = fontsize)
    plt.savefig(outroot+'8_local_metric_PCA.pdf')
    plt.show()

    ###############################################################################
    # PLOT GEODESICS
    ###############################################################################
