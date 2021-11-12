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
from getdist import plots
import scipy
import matplotlib.pyplot as plt
import seaborn as sns

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
# import the tensiometer tools that we need:
from tensiometer import utilities
import synthetic_probability
from tensorflow_probability import bijectors as bj

###############################################################################
# initial settings:

# output folder:
out_folder = './results/example_DES_shear/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

###############################################################################
# import chains:

prior_chain = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'002_DESY1_shear_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'002_DESY1_shear', no_cache=True, settings=DES_generate.settings)

###############################################################################
# process chains:

# add log parameters:
for ch in [prior_chain, posterior_chain]:
    temp_names = ch.getParamNames().list()
    for name in temp_names:
        if np.all(ch.samples[:, ch.index[name]] > 0.):
            ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log '+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()

###############################################################################
# train the relevant flows:

param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns']
params_flow_cache = out_folder+'params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain, posterior_chain, params_flow_cache)
params_prior_flow, params_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
log_params_prior_flow = synthetic_probability.TransformedDiffFlowCallback(params_prior_flow, transformation)
log_params_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(params_posterior_flow, transformation)
log_param_names = log_params_prior_flow.param_names

###############################################################################
# sanity triangle plot:

if __name__ == '__main__':

    num_samples = 100000

    # prior and posterior chain:
    g = plots.get_subplot_plotter()
    g.triangle_plot([prior_chain,
                     log_params_prior_flow.MCSamples(num_samples),
                     posterior_chain,
                     log_params_posterior_flow.MCSamples(num_samples)],
                    params=log_param_names, filled=False)
    g.export(out_folder+'/0_sample_prior_posterior_distribution_log.pdf')

    g = plots.get_subplot_plotter()
    g.triangle_plot([prior_chain,
                     params_prior_flow.MCSamples(num_samples),
                     posterior_chain,
                     params_posterior_flow.MCSamples(num_samples)],
                    params=param_names, filled=False)
    g.export(out_folder+'/0_sample_prior_posterior_distribution.pdf')

###############################################################################
# PCA of covariance:

if __name__ == '__main__':

    num_params = len(log_param_names)
    # compute covariance and PCA of fisher:
    covariance = posterior_chain.cov(log_param_names)
    fisher = np.linalg.inv(covariance)
    eig, eigv = np.linalg.eigh(fisher)
    sqrt_fisher = scipy.linalg.sqrtm(fisher)
    # sort modes:
    idx = np.argsort(eig)[::-1]
    eig = eig[idx]
    eigv = eigv[:, idx]
    # compute contributions:
    temp = np.dot(sqrt_fisher, eigv)
    contributions = temp * temp / eig

    # plot contributions:
    plt.figure(figsize=(num_params*1, num_params*1))
    im1 = plt.imshow(contributions, cmap='viridis')
    for i in range(num_params):
        for j in range(num_params):
            if contributions[j, i] > 0.5:
                col = 'k'
            else:
                col = 'w'
            plt.text(i, j, np.round(contributions[j, i], 2), va='center', ha='center', color=col)
    plt.xlabel('PCA mode (value)')
    plt.ylabel('Parameters')
    ticks = np.arange(num_params)
    labels = [str(t+1)+'\n ('+str(l)+')' for t, l in zip(ticks, np.round(np.sqrt(eig), 2))]
    plt.xticks(ticks, labels, horizontalalignment='center')
    labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in log_param_names]
    plt.yticks(ticks, labels, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_folder+'/1_PCA_mode_contributions.pdf')
    plt.close('all')

    # print out modes:
    for i in range(num_params):
        idx_max = np.argmax(contributions[:, i])
        print('* Mode', i+1)
        print('  Sqrt eig = ', np.round(np.sqrt(eig[i]),2))
        _norm_eigv = eigv[:, i] / eigv[idx_max, i]
        with np.printoptions(precision=2, suppress=True):
            print('  Variance contributions', contributions[:, i])
        string = ''
        for j in range(num_params):
            _name = log_param_names[j]
            _mean = posterior_chain.getMeans([posterior_chain.index[log_param_names[j]]])[0]
            _mean = '{0:+}'.format(np.round(-_mean, 2))
            _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
            string += _temp+'*('+_name+' '+_mean+') '
        print('  Mode =', string, '= 0')
        print(' ')

    # plot triangle with lines:
    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain], params=log_param_names, filled=True)
    # add the modes:
    for i in range(num_params-1):
        for j in range(i+1, num_params):
            ax = g.subplots[j, i]
            # get mean:
            m1, m2 = posterior_chain.getMeans(pars=[posterior_chain.index[name]
                                              for name in [log_param_names[i], log_param_names[j]]])
            ax.scatter(m1, m2, color='k')
            alpha = 3.*np.linspace(-1., 1., 100)
            for k in range(num_params):
                ax.axline([m1, m2], [m1 + eigv[i, k], m2 + eigv[j, k]], color=sns.hls_palette(num_params)[k], label='Mode '+str(k+1))
    g.fig.legend(*ax.get_legend_handles_labels())
    g.export(out_folder+'/2_PCA_triangle.pdf')

###############################################################################
# PCA of correlation:

if __name__ == '__main__':

    num_params = len(log_param_names)
    # compute covariance and PCA of fisher:
    covariance = posterior_chain.cov(log_param_names)
    fisher = np.linalg.inv(covariance)
    weights = np.diag(1./np.sqrt(np.diag(fisher)))
    fisher_correlation = np.dot(np.dot(weights, fisher), weights)
    eig, eigv = np.linalg.eigh(fisher_correlation)
    sqrt_fisher = scipy.linalg.sqrtm(fisher_correlation)

    # sort modes:
    idx = np.argsort(eig)[::-1]
    eig = eig[idx]
    eigv = eigv[:, idx]
    # compute contributions:
    temp = np.dot(sqrt_fisher, eigv)
    contributions = temp * temp / eig

    # plot contributions:
    plt.figure(figsize=(num_params*1, num_params*1))
    im1 = plt.imshow(contributions, cmap='viridis')
    for i in range(num_params):
        for j in range(num_params):
            if contributions[j, i] > 0.5:
                col = 'k'
            else:
                col = 'w'
            plt.text(i, j, np.round(contributions[j, i], 2), va='center', ha='center', color=col)
    plt.xlabel('PCA mode (value)')
    plt.ylabel('Parameters')
    ticks = np.arange(num_params)
    labels = [str(t+1)+'\n ('+str(l)+')' for t, l in zip(ticks, np.round(np.sqrt(eig), 2))]
    plt.xticks(ticks, labels, horizontalalignment='center')
    labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in log_param_names]
    plt.yticks(ticks, labels, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_folder+'/3_PCCA_mode_contributions.pdf')
    plt.close('all')

    # print out modes:
    for i in range(num_params):
        idx_max = np.argmax(contributions[:, i])
        print('* Mode', i+1)
        print('  Sqrt eig = ', np.round(np.sqrt(eig[i]),2))
        _norm_eigv = eigv[:, i] / eigv[idx_max, i] * weights[idx_max, idx_max] / np.diag(weights)
        with np.printoptions(precision=2, suppress=True):
            print('  Variance contributions', contributions[:, i])
        string = ''
        for j in range(num_params):
            _name = log_param_names[j]
            _mean = posterior_chain.getMeans([posterior_chain.index[log_param_names[j]]])[0]
            _mean = '{0:+}'.format(np.round(-_mean, 2))
            _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
            string += _temp+'*('+_name+' '+_mean+') '
        print('  Mode =', string, '= 0')
        print(' ')

    # plot triangle with lines:
    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain], params=log_param_names, filled=True)
    # add the modes:
    for i in range(num_params-1):
        for j in range(i+1, num_params):
            ax = g.subplots[j, i]
            # get mean:
            m1, m2 = posterior_chain.getMeans(pars=[posterior_chain.index[name]
                                              for name in [log_param_names[i], log_param_names[j]]])
            ax.scatter(m1, m2, color='k')
            alpha = 3.*np.linspace(-1., 1., 100)
            for k in range(num_params):
                _direction = eigv[:, k] / np.diag(weights)
                _direction = _direction / np.sqrt(np.dot(_direction, _direction))
                ax.axline([m1, m2], [m1 + _direction[i], m2 + _direction[j]], color=sns.hls_palette(num_params)[k], label='Mode '+str(k+1))
    g.fig.legend(*ax.get_legend_handles_labels())
    g.export(out_folder+'/4_PCCA_triangle.pdf')

###############################################################################
# KL decomposition:

if __name__ == '__main__':

    num_params = len(log_param_names)
    covariance = posterior_chain.cov(log_param_names)
    prior_covariance = prior_chain.cov(log_param_names)

    fisher = np.linalg.inv(covariance)
    prior_fisher = np.linalg.inv(prior_covariance)
    eig, eigv = utilities.KL_decomposition(fisher, prior_fisher)
    sqrt_fisher = scipy.linalg.sqrtm(fisher)

    # sort modes:
    idx = np.argsort(eig)[::-1]
    eig = eig[idx]
    eigv = eigv[:, idx]

    # compute contributions:
    temp = np.dot(sqrt_fisher, eigv)
    contributions = temp * temp / eig

    # plot contributions:
    plt.figure(figsize=(num_params*1, num_params*1))
    im1 = plt.imshow(contributions, cmap='viridis')
    for i in range(num_params):
        for j in range(num_params):
            if contributions[j, i] > 0.5:
                col = 'k'
            else:
                col = 'w'
            plt.text(i, j, np.round(contributions[j, i], 2), va='center', ha='center', color=col)
    plt.xlabel('KL mode (value)')
    plt.ylabel('Parameters')
    ticks = np.arange(num_params)
    labels = [str(t+1)+'\n ('+str(l)+')' for t, l in zip(ticks, np.round(np.sqrt(eig), 2))]
    plt.xticks(ticks, labels, horizontalalignment='center')
    labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in log_param_names]
    plt.yticks(ticks, labels, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_folder+'/5_KL_mode_contributions.pdf')
    plt.close('all')

    # print out modes:
    for i in range(num_params):
        idx_max = np.argmax(contributions[:, i])
        print('* Mode', i+1)
        print('  Sqrt eig = ', np.round(np.sqrt(eig[i]),2))
        _directions = np.linalg.inv(eigv).T
        _norm_eigv = _directions[:, i] / _directions[idx_max, i]
        with np.printoptions(precision=2, suppress=True):
            print('  Variance contributions', contributions[:, i])
        string = ''
        for j in range(num_params):
            _name = log_param_names[j]
            _mean = posterior_chain.getMeans([posterior_chain.index[log_param_names[j]]])[0]
            _mean = '{0:+}'.format(np.round(-_mean, 2))
            _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
            string += _temp+'*('+_name+' '+_mean+') '
        print('  Mode =', string, '= 0')
        print(' ')

    # plot triangle with lines:
    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain], params=log_param_names, filled=True)
    # add the modes:
    for i in range(num_params-1):
        for j in range(i+1, num_params):
            ax = g.subplots[j, i]
            # get mean:
            m1, m2 = posterior_chain.getMeans(pars=[posterior_chain.index[name]
                                              for name in [log_param_names[i], log_param_names[j]]])
            ax.scatter(m1, m2, color='k')
            alpha = 3.*np.linspace(-1., 1., 100)
            for k in range(num_params):
                _direction = eigv[:, k]
                _direction = _direction / np.sqrt(np.dot(_direction, _direction))
                ax.axline([m1, m2], [m1 + _direction[i], m2 + _direction[j]], color=sns.hls_palette(num_params)[k], label='Mode '+str(k+1))
    g.fig.legend(*ax.get_legend_handles_labels())
    g.export(out_folder+'/6_KL_triangle.pdf')

###############################################################################
# KL of local covariance:

if __name__ == '__main__':

    num_params = len(log_param_names)
    # find MAP of data chain:
    reference_coords = log_params_posterior_flow.sample_MAP
    reference_coords = posterior_chain.getMeans(pars=[posterior_chain.index[name] for name in log_param_names])
    # get local fisher:
    fisher = log_params_posterior_flow.metric(log_params_posterior_flow.cast([reference_coords]))[0]
    prior_fisher = log_params_prior_flow.metric(log_params_posterior_flow.cast([reference_coords]))[0]
    eig, eigv = utilities.KL_decomposition(fisher, prior_fisher)
    sqrt_fisher = scipy.linalg.sqrtm(fisher)

    # sort modes:
    idx = np.argsort(eig)[::-1]
    eig = eig[idx]
    eigv = eigv[:, idx]

    # compute contributions:
    temp = np.dot(sqrt_fisher, eigv)
    contributions = temp * temp / eig

    # plot contributions:
    plt.figure(figsize=(num_params*1, num_params*1))
    im1 = plt.imshow(contributions, cmap='viridis')
    for i in range(num_params):
        for j in range(num_params):
            if contributions[j, i] > 0.5:
                col = 'k'
            else:
                col = 'w'
            plt.text(i, j, np.round(contributions[j, i], 2), va='center', ha='center', color=col)
    plt.xlabel('KL mode (value)')
    plt.ylabel('Parameters')
    ticks = np.arange(num_params)
    labels = [str(t+1)+'\n ('+str(l)+')' for t, l in zip(ticks, np.round(np.sqrt(eig), 2))]
    plt.xticks(ticks, labels, horizontalalignment='center')
    labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in log_param_names]
    plt.yticks(ticks, labels, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_folder+'/7_LKL_mode_contributions.pdf')
    plt.close('all')

    # print out modes:
    for i in range(num_params):
        idx_max = np.argmax(contributions[:, i])
        print('* Mode', i+1)
        print('  Sqrt eig = ', np.round(np.sqrt(eig[i]),2))
        _directions = np.linalg.inv(eigv).T
        _norm_eigv = _directions[:, i] / _directions[idx_max, i]
        with np.printoptions(precision=2, suppress=True):
            print('  Variance contributions', contributions[:, i])
        string = ''
        for j in range(num_params):
            _name = log_param_names[j]
            _mean = posterior_chain.getMeans([posterior_chain.index[log_param_names[j]]])[0]
            _mean = '{0:+}'.format(np.round(-_mean, 2))
            _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
            string += _temp+'*('+_name+' '+_mean+') '
        print('  Mode =', string, '= 0')
        print(' ')

    # plot triangle with lines:
    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain], params=log_param_names, filled=True)
    # add the modes:
    for i in range(num_params-1):
        for j in range(i+1, num_params):
            ax = g.subplots[j, i]
            # get mean:
            m1, m2 = posterior_chain.getMeans(pars=[posterior_chain.index[name]
                                              for name in [log_param_names[i], log_param_names[j]]])
            ax.scatter(m1, m2, color='k')
            alpha = 3.*np.linspace(-1., 1., 100)
            for k in range(num_params):
                _direction = eigv[:, k]
                _direction = _direction / np.sqrt(np.dot(_direction, _direction))
                ax.axline([m1, m2], [m1 + _direction[i], m2 + _direction[j]], color=sns.hls_palette(num_params)[k], label='Mode '+str(k+1))
    g.fig.legend(*ax.get_legend_handles_labels())
    g.export(out_folder+'/8_LKL_triangle.pdf')
# print(reference_coords)
###############################################################################
# PCA of local covariance:

if __name__ == '__main__':

    num_params = len(log_param_names)
    # compute local fisher and PCA of fisher:

    fisher = log_params_posterior_flow.metric(log_params_posterior_flow.cast([reference_coords]))[0]
    eig, eigv = np.linalg.eigh(fisher)
    sqrt_fisher = scipy.linalg.sqrtm(fisher)
    # sort modes:
    idx = np.argsort(eig)[::-1]
    eig = eig[idx]
    eigv = eigv[:, idx]
    # compute contributions:
    temp = np.dot(sqrt_fisher, eigv)
    contributions = temp * temp / eig

    # plot contributions:
    plt.figure(figsize=(num_params*1, num_params*1))
    im1 = plt.imshow(contributions, cmap='viridis')
    for i in range(num_params):
        for j in range(num_params):
            if contributions[j, i] > 0.5:
                col = 'k'
            else:
                col = 'w'
            plt.text(i, j, np.round(contributions[j, i], 2), va='center', ha='center', color=col)
    plt.xlabel('PCA mode (value)')
    plt.ylabel('Parameters')
    ticks = np.arange(num_params)
    labels = [str(t+1)+'\n ('+str(l)+')' for t, l in zip(ticks, np.round(np.sqrt(eig), 2))]
    plt.xticks(ticks, labels, horizontalalignment='center')
    labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in log_param_names]
    plt.yticks(ticks, labels, horizontalalignment='right')
    plt.tight_layout()
    plt.savefig(out_folder+'/9_LPCA_mode_contributions.pdf')
    plt.close('all')

    # print out modes:
    for i in range(num_params):
        idx_max = np.argmax(contributions[:, i])
        print('* Mode', i+1)
        print('  Sqrt eig = ', np.round(np.sqrt(eig[i]),2))
        _norm_eigv = eigv[:, i] / eigv[idx_max, i]
        with np.printoptions(precision=2, suppress=True):
            print('  Variance contributions', contributions[:, i])
        string = ''
        for j in range(num_params):
            _name = log_param_names[j]
            _mean = posterior_chain.getMeans([posterior_chain.index[log_param_names[j]]])[0]
            _mean = '{0:+}'.format(np.round(-_mean, 2))
            _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
            string += _temp+'*('+_name+' '+_mean+') '
        print('  Mode =', string, '= 0')
        print(' ')

    # plot triangle with lines:
    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain], params=log_param_names, filled=True)
    # add the modes:
    for i in range(num_params-1):
        for j in range(i+1, num_params):
            ax = g.subplots[j, i]
            # get mean:
            m1, m2 = posterior_chain.getMeans(pars=[posterior_chain.index[name]
                                              for name in [log_param_names[i], log_param_names[j]]])
            ax.scatter(m1, m2, color='k')
            alpha = 3.*np.linspace(-1., 1., 100)
            for k in range(num_params):
                ax.axline([m1, m2], [m1 + eigv[i, k], m2 + eigv[j, k]], color=sns.hls_palette(num_params)[k], label='Mode '+str(k+1))
    g.fig.legend(*ax.get_legend_handles_labels())
    g.export(out_folder+'/10_LPCA_triangle.pdf')
