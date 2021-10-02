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
            ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log'+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()

###############################################################################
# decide parameters to use:

param_names = ['log_omegam', 'log_omegab', 'log_H0', 'log_sigma8', 'ns',
               'log_DES_b1', 'log_DES_b2', 'log_DES_b3', 'log_DES_b4', 'log_DES_b5',
               'DES_m1', 'DES_m2', 'DES_m3', 'DES_m4',
               'DES_AIA', 'DES_alphaIA',
               'DES_DzL1', 'DES_DzL2', 'DES_DzL3', 'DES_DzL4', 'DES_DzL5',
               'DES_DzS1', 'DES_DzS2', 'DES_DzS3', 'DES_DzS4']
param_names = ['log_omegam', 'log_omegab', 'log_sigma8']
param_names = ['log_omegam', 'log_sigma8']
param_names = ['log_omegam', 'log_omegab', 'log_H0', 'log_sigma8', 'log_ns']
param_names = ['log_omegam', 'log_omegab', 'log_H0', 'log_sigma8', 'ns']

num_params = len(param_names)

###############################################################################
# sanity triangle plot:

g = plots.get_subplot_plotter()
g.triangle_plot([prior_chain, posterior_chain], params=param_names, filled=False)
g.export(out_folder+'/0_prior_posterior_distribution.pdf')

###############################################################################
# PCA of covariance:

# compute covariance and PCA of fisher:
covariance = posterior_chain.cov(param_names)
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
labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in param_names]
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
        _name = param_names[j]
        _mean = posterior_chain.getMeans([posterior_chain.index[param_names[j]]])[0]
        _mean = '{0:+}'.format(np.round(-_mean, 2))
        _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
        string += _temp+'*('+_name+' '+_mean+') '
    print('  Mode =', string, '= 0')
    print(' ')

# plot triangle with lines:
g = plots.get_subplot_plotter()
g.triangle_plot([posterior_chain], params=param_names, filled=True)
# add the modes:
for i in range(num_params-1):
    for j in range(i+1, num_params):
        ax = g.subplots[j, i]
        # get mean:
        m1, m2 = posterior_chain.getMeans(pars=[posterior_chain.index[name]
                                          for name in [param_names[i], param_names[j]]])
        ax.scatter(m1, m2, color='k')
        alpha = 3.*np.linspace(-1., 1., 100)
        for k in range(num_params):
            ax.axline([m1, m2], [m1 + eigv[i, k], m2 + eigv[j, k]], color=sns.hls_palette(num_params)[k], label='Mode '+str(k+1))
g.fig.legend(*ax.get_legend_handles_labels())
g.export(out_folder+'/2_PCA_triangle.pdf')

###############################################################################
# PCA of correlation:

# compute covariance and PCA of fisher:
covariance = posterior_chain.cov(param_names)
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
labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in param_names]
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
        _name = param_names[j]
        _mean = posterior_chain.getMeans([posterior_chain.index[param_names[j]]])[0]
        _mean = '{0:+}'.format(np.round(-_mean, 2))
        _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
        string += _temp+'*('+_name+' '+_mean+') '
    print('  Mode =', string, '= 0')
    print(' ')

# plot triangle with lines:
g = plots.get_subplot_plotter()
g.triangle_plot([posterior_chain], params=param_names, filled=True)
# add the modes:
for i in range(num_params-1):
    for j in range(i+1, num_params):
        ax = g.subplots[j, i]
        # get mean:
        m1, m2 = posterior_chain.getMeans(pars=[posterior_chain.index[name]
                                          for name in [param_names[i], param_names[j]]])
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

covariance = posterior_chain.cov(param_names)
prior_covariance = prior_chain.cov(param_names)

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
plt.xlabel('PCA mode (value)')
plt.ylabel('Parameters')
ticks = np.arange(num_params)
labels = [str(t+1)+'\n ('+str(l)+')' for t, l in zip(ticks, np.round(np.sqrt(eig), 2))]
plt.xticks(ticks, labels, horizontalalignment='center')
labels = ['$'+posterior_chain.getParamNames().parWithName(name).label+'$' for name in param_names]
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
        _name = param_names[j]
        _mean = posterior_chain.getMeans([posterior_chain.index[param_names[j]]])[0]
        _mean = '{0:+}'.format(np.round(-_mean, 2))
        _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
        string += _temp+'*('+_name+' '+_mean+') '
    print('  Mode =', string, '= 0')
    print(' ')

# plot triangle with lines:
g = plots.get_subplot_plotter()
g.triangle_plot([posterior_chain], params=param_names, filled=True)
# add the modes:
for i in range(num_params-1):
    for j in range(i+1, num_params):
        ax = g.subplots[j, i]
        # get mean:
        m1, m2 = posterior_chain.getMeans(pars=[posterior_chain.index[name]
                                          for name in [param_names[i], param_names[j]]])
        ax.scatter(m1, m2, color='k')
        alpha = 3.*np.linspace(-1., 1., 100)
        for k in range(num_params):
            _direction = eigv[:, k]
            _direction = _direction / np.sqrt(np.dot(_direction, _direction))
            ax.axline([m1, m2], [m1 + _direction[i], m2 + _direction[j]], color=sns.hls_palette(num_params)[k], label='Mode '+str(k+1))
g.fig.legend(*ax.get_legend_handles_labels())
g.export(out_folder+'/6_PCCA_triangle.pdf')
