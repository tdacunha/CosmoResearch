# -*- coding: utf-8 -*-

"""
Video of training
"""

###############################################################################
# initial imports:
import os
import numpy as np

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import tensiometer.gaussian_tension as gaussian_tension
from scipy.integrate import simps

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import synthetic_probability
from tensorflow.keras.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau()]

from tensiometer import utilities
import example_2_generate as example
import analyze_2d_example
import scipy.stats

import tensorflow as tf

###############################################################################
# initial settings:

# output folder:
out_folder = './results/example_2/video/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# number of samples:
n_samples = 1000000

# plotting preferences:
figsize = (8, 8)
fontsize = 15
levels = [utilities.from_sigma_to_confidence(i) for i in range(4, 1, -1)]

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

###############################################################################
# set up the flow:

# posterior:
num_params = len(example.param_names)
n_maf = 4*num_params
hidden_units = [num_params*4]*3
batch_size = 2*8192
epochs = 80
steps_per_epoch = 128

# initialize flow:
posterior_flow = synthetic_probability.DiffFlowCallback(example.posterior_chain,
                                                        param_ranges=example.param_ranges, param_names=example.posterior_chain.getParamNames().list(),
                                                        feedback=0, learning_rate=0.01, n_maf=n_maf, hidden_units=hidden_units)

# plot during training:
param_ranges = [[0.01, 0.7-0.01], [0.01, 1.7-0.01]]
param_names = example.posterior_chain.getParamNames().list()

# parameter grids:
P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 200)
P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 200)
x, y = P1, P2
X, Y = np.meshgrid(x, y)

# obtain posterior on grid from samples:
density = example.posterior_chain.get2DDensity(param_names[0], param_names[1], normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
density.P = density.P / simps(simps(density.P, density.y), density.x)

# run the frame loop:
for ind in range(110):

    # feedback:
    print('Frame:', ind)

    # train:
    if ind > 0:
        if ind < 50:
            steps_per_epoch = 1
        else:
            steps_per_epoch = 100
        posterior_flow.train(batch_size=batch_size, epochs=1, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    # create figure:
    plt.figure(figsize=(2*figsize[0], figsize[1]))
    gs = gridspec.GridSpec(1, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax2 = plt.subplot(gs[0, 1])

    # plot contours:
    ax1.contour(_X, _Y, density.P, analyze_2d_example.get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels])
    origin = [0, 0]
    theta = np.linspace(0.0, 2.*np.pi, 200)
    for i in range(4):
        _length = np.sqrt(scipy.stats.chi2.isf(1.-utilities.from_sigma_to_confidence(i), 2))
        ax2.plot(origin[0]+_length*np.sin(theta), origin[1]+_length*np.cos(theta), ls='-', lw=1., color='k')

    # compute flow probability on a grid:
    log_P = posterior_flow.log_probability(np.array([X, Y], dtype=np.float32).T)
    log_P = np.array(log_P).T
    P = np.exp(log_P)
    P = P / simps(simps(P, y), x)
    ax1.contour(X, Y, P, analyze_2d_example.get_levels(P, x, y, levels), linewidths=1., linestyles='--', colors=['red' for i in levels])

    # plot samples:
    num_samples = 1000
    tot_num_samples = len(example.posterior_chain.samples)
    temp_samples = example.posterior_chain.samples[::tot_num_samples // num_samples, :]
    abs_temp_samples = posterior_flow.map_to_abstract_coord(posterior_flow.cast(temp_samples)).numpy()

    ax1.scatter(*temp_samples.T, s=0.8, c=temp_samples[:, 0])
    ax2.scatter(*abs_temp_samples.T, s=0.8, c=temp_samples[:, 0])

    # finalize plot:
    ax1.set_xlabel('$\\theta_1$', fontsize=fontsize)
    ax1.set_ylabel('$\\theta_2$', fontsize=fontsize)
    ax2.set_xlabel('$Z_1$', fontsize=fontsize)
    ax2.set_ylabel('$Z_2$', fontsize=fontsize)

    ax1.set_xlim([0.0, 0.5])
    ax1.set_ylim([0.5, 1.5])
    ax2.set_xlim([-5, 5])
    ax2.set_ylim([-5, 5])

    # title:
    ax1.text(0.01, 1.02, 'a) Parameter space', verticalalignment='bottom', horizontalalignment='left', fontsize=fontsize, transform=ax1.transAxes)
    ax2.text(0.01, 1.02, 'b) Abstract space', verticalalignment='bottom', horizontalalignment='left', fontsize=fontsize, transform=ax2.transAxes)

    # update dimensions:
    bottom = 0.1
    top = 0.94
    left = 0.05
    right = 0.99
    wspace = 0.1
    hspace = 0.2
    gs.update(bottom=bottom, top=top, left=left, right=right, wspace=wspace, hspace=hspace)

    plt.savefig(out_folder+f'/figure_{ind:05d}.png', dpi=300)
    plt.close('all')
