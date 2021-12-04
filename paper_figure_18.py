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
from tensiometer import utilities, gaussian_tension

# import example:
import example_DES_Y1 as example

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

prior_chain = example.posterior_chain_lcdm_shear
posterior_chain = example.posterior_chain_lcdm_3x2

prior_flow = example.lcdm_shear_log_params_full_posterior_flow
posterior_flow = example.lcdm_3x2_log_params_shear_posterior_flow

###############################################################################
# decide parameters to use:

param_names = posterior_flow.param_names
num_params = len(param_names)

###############################################################################
# Compute KL contributions:

# reference point:
reference_shear = posterior_chain.getMeans(pars=[posterior_chain.index[name] for name in param_names])
# local fisher:
fisher = posterior_flow.metric(posterior_flow.cast([reference_shear]))[0]
prior_fisher = prior_flow.metric(prior_flow.cast([reference_shear]))[0]
# global fisher:
#fisher = np.linalg.inv(example_shear.posterior_chain.cov(example_shear.log_param_names))
#prior_fisher = np.linalg.inv(example_shear.prior_chain.cov(example_shear.log_param_names))

eig, eigv = utilities.KL_decomposition(fisher, prior_fisher)
sqrt_fisher = scipy.linalg.sqrtm(fisher)

# sort modes:
idx = np.argsort(eig)[::-1]
eig = eig[idx]
eigv = eigv[:, idx]
# compute contributions:
temp = np.dot(sqrt_fisher, eigv)
contributions_1 = temp * temp / eig
contributions_1 = np.flip(contributions_1, axis=0)
eig_1 = eig.copy()

###############################################################################
# Make the plot:

x_size = 8.54
y_size = 8.0
main_fontsize = 10.0

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])

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

# set ticks:
for ax in [ax1]:
    ax.set_xticks(range(num_params))
    ax.set_yticks(range(num_params))
_labels = ['$\\alpha_{\\rm IA}$',
           '$A_{\\rm IA}$',
           '$\\log n_s$',
           '$\\log H_0$',
           '$\\log \\Omega_b$',
           '$\\log \\sigma_8$',
           '$\\log \\Omega_m$']
ax1.set_yticklabels(_labels, fontsize=0.9*main_fontsize)

_temp = eig_1 - 1.
_temp[_temp < 0.] = 0.
ax1.set_xticklabels([str(t+1)+'\n ('+str(l)+')' for t, l in zip(range(num_params), np.round(np.sqrt(_temp), 2))], fontsize=0.9*main_fontsize)

# axes labels:
ax1.set_xlabel('CPC mode $(\\sqrt{\\lambda-1})$', fontsize=main_fontsize);
ax1.set_ylabel('Parameter', fontsize=main_fontsize);

# update dimensions:
bottom = 0.19
top = 0.99
left = 0.20
right = 0.99
wspace = 0.03
hspace = 0.08
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)

plt.savefig(out_folder+'/figure_18.pdf')
plt.close('all')
