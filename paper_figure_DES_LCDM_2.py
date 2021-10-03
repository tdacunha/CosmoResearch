# -*- coding: utf-8 -*-

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
import matplotlib.gridspec as gridspec

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
# import the tensiometer tools that we need:
from tensiometer import utilities

###############################################################################
# initial settings:

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

###############################################################################
# import chains:

prior_chain_1 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'002_DESY1_shear_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_1 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'002_DESY1_shear', no_cache=True, settings=DES_generate.settings)

prior_chain_2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'001_DESY1_3x2_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_2 = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'001_DESY1_3x2', no_cache=True, settings=DES_generate.settings)

###############################################################################
# process chains:

# add log parameters:
for ch in [prior_chain_1, posterior_chain_1, prior_chain_2, posterior_chain_2]:
    temp_names = ch.getParamNames().list()
    for name in temp_names:
        if np.all(ch.samples[:, ch.index[name]] > 0.):
            ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log '+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()

###############################################################################
# decide parameters to use:

param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'ns'][::-1]
num_params = len(param_names)

###############################################################################
# Compute PCA contributions:

# compute covariance and PCA of fisher:
covariance = posterior_chain_1.cov(param_names)
prior_covariance = prior_chain_1.cov(param_names)
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
contributions_1 = temp * temp / eig
eig_1 = eig.copy()

# compute covariance and PCA of fisher:
covariance = posterior_chain_2.cov(param_names)
prior_covariance = prior_chain_2.cov(param_names)
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
contributions_2 = temp * temp / eig
eig_2 = eig.copy()

###############################################################################
# Make the plot:

x_size = 18.0
y_size = 9.0
main_fontsize = 10.0

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

X, Y = np.meshgrid(np.arange(num_params), np.arange(num_params))

# plot the matrices:
#im1 = ax1.imshow(contributions_1, vmin=0.0, vmax=1.0, cmap='Blues')
im1 = ax1.pcolormesh(X, Y, contributions_1, linewidth=0, rasterized=True, shading='auto', vmin=0.0, vmax=1.0, cmap='Blues')
for i in range(num_params):
    for j in range(num_params):
        if contributions_1[j, i] > 0.5:
            col = 'w'
        else:
            col = 'k'
        ax1.text(i, j, np.round(contributions_1[j, i], 2), va='center', ha='center', color=col)

#im2 = ax2.imshow(contributions_2, vmin=0.0, vmax=1.0, cmap='Blues')
im2 = ax2.pcolormesh(X, Y, contributions_2, linewidth=0, rasterized=True, shading='auto', vmin=0.0, vmax=1.0, cmap='Blues')
for i in range(num_params):
    for j in range(num_params):
        if contributions_2[j, i] > 0.5:
            col = 'w'
        else:
            col = 'k'
        ax2.text(i, j, np.round(contributions_2[j, i], 2), va='center', ha='center', color=col)

# set ticks:
for ax in [ax1, ax2]:
    ax.set_xticks(range(num_params))
    ax.set_yticks(range(num_params))
ax1.set_yticklabels([r'$'+name.label+'$' for name in posterior_chain_1.getParamNames().parsWithNames(param_names)], fontsize=0.9*main_fontsize)
ax2.set_yticklabels([])
_temp = eig_1 - 1.
_temp[_temp < 0.] = 0.
ax1.set_xticklabels([str(t+1)+'\n ('+str(l)+')' for t, l in zip(range(num_params), np.round(np.sqrt(_temp), 2))], fontsize=0.9*main_fontsize)
_temp = eig_2 - 1.
_temp[_temp < 0.] = 0.
ax2.set_xticklabels([str(t+1)+'\n ('+str(l)+')' for t, l in zip(range(num_params), np.round(np.sqrt(_temp), 2))], fontsize=0.9*main_fontsize)

# axes labels:
ax1.set_xlabel('KL mode $(\\sqrt{\\lambda-1})$', fontsize=main_fontsize);
ax2.set_xlabel('KL mode $(\\sqrt{\\lambda-1})$', fontsize=main_fontsize);
ax1.set_ylabel('Parameter', fontsize=main_fontsize);

# title:
ax1.text(0.01, 1.03, 'a) DES Y1 shear', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax1.transAxes)
ax2.text(0.01, 1.03, 'b) DES Y1 3x2', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax2.transAxes)

# update dimensions:
bottom = 0.19
top = 0.92
left = 0.11
right = 0.99
wspace = 0.03
hspace = 0.08
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)

plt.savefig(out_folder+'/figure_DES_LCDM_2.pdf')
plt.close('all')
