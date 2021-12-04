# -*- coding: utf-8 -*-

###############################################################################
# initial imports:

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.integrate import simps
from getdist import plots, MCSamples
import color_utilities
import getdist
getdist.chains.print_load_details = False
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import tensorflow as tf

# add path for correct version of tensiometer:
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
from tensiometer import utilities

###############################################################################
# initial settings:

import example_2_generate as example
import analyze_2d_example

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)


###############################################################################
# plot:

# plot size in cm. Has to match to draft to make sure font sizes are consistent
x_size = 8.54
y_size = 7.0
main_fontsize = 10.0

levels = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]
param_ranges = [[0.01, 0.6], [0.4, 1.5]]

# define the grid:
P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 400)
P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 400)
x, y = P1, P2
X, Y = np.meshgrid(x, y)

# compute probability:
log_P = example.posterior_flow.log_probability(np.array([X, Y], dtype=np.float32).T)
log_P = np.array(log_P).T
P = np.exp(log_P)
P = P / simps(simps(P, y), x)

# compute maximum posterior and metric:
maximum_posterior = example.posterior_flow.MAP_coord

# compute local fisher:
coarse_P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 20)
coarse_P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 20)
coarse_x, coarse_y = coarse_P1, coarse_P2
coarse_X, coarse_Y = np.meshgrid(coarse_x, coarse_y)
# restructure meshgrid of points to give an array of coordinates
coords = np.array([coarse_X, coarse_Y], dtype=np.float32).reshape(2, -1).T
# compute the metric at all coordinates
local_metrics = example.posterior_flow.metric(coords)
# compute the PCA eigenvalues and eigenvectors of each local metric
PCA_eig, PCA_eigv = np.linalg.eigh(local_metrics)
# sort PCA so first mode is index 0
idx = np.argsort(PCA_eig, axis=1)[0]
PCA_eig = PCA_eig[:, idx]
PCA_eigv = PCA_eigv[:, :, idx]

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])

ax1.contour(X, Y, P, analyze_2d_example.get_levels(P, x, y, levels), linewidths=1., zorder=-1., linestyles='-', colors=[color_utilities.nice_colors(6) for i in levels])

# plot arrows:
mode = 0
ax1.quiver(coords[:, 0], coords[:, 1], PCA_eigv[:, 0, mode], PCA_eigv[:, 1, mode],
           color=color_utilities.nice_colors(0), linewidths=1., angles='xy', label='First mode',
           headlength=0., headaxislength=0, headwidth=1, pivot='mid', width=0.005, scale=20.0)
mode = 1
ax1.quiver(coords[:, 0], coords[:, 1], PCA_eigv[:, 0, mode], PCA_eigv[:, 1, mode],
           color=color_utilities.nice_colors(1), linewidths=1., angles='xy', label='Second mode',
           headlength=0., headaxislength=0, headwidth=1, pivot='mid', width=0.005, scale=20.0)

# limits:
ax1.set_xlim([0.1, 0.45])
ax1.set_ylim([0.6, 1.2])

# ticks:
ticks = [0.1, 0.2, 0.3, 0.4, 0.45]
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_xticklabels()[0].set_horizontalalignment('left')
ax1.get_xticklabels()[-1].set_horizontalalignment('right')

ticks = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
ax1.set_yticks(ticks)
ax1.set_yticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_yticklabels()[0].set_verticalalignment('bottom')
ax1.get_yticklabels()[-1].set_verticalalignment('top')

# axes labels:
ax1.set_xlabel(r'$\theta_1$', fontsize=main_fontsize);
ax1.set_ylabel(r'$\theta_2$', fontsize=main_fontsize);

# legend:
leg_handlers = [mlines.Line2D([], [], lw=1., ls='-', color='k'),
                mlines.Line2D([], [], lw=1., ls='-', color=color_utilities.nice_colors(1)),
                mlines.Line2D([], [], lw=1., ls='-', color=color_utilities.nice_colors(0)),
                ]
legend_labels = [r'$\mathcal{P}$',
                 'Local PC 1',
                 'Local PC 2']

leg = fig.legend(handles=leg_handlers,
                labels=legend_labels,
                fontsize=0.9*main_fontsize,
                frameon=True,
                fancybox=False,
                edgecolor='k',
                ncol=len(legend_labels),
                borderaxespad=0.0,
                columnspacing=2.0,
                handlelength=1.5,
                handletextpad=0.3,
                loc = 'lower center',
                bbox_to_anchor=(0.0, 0.02, 1.2, 0.9),
                )
leg.get_frame().set_linewidth('0.8')

# update dimensions:
bottom = .26
top = 0.99
left = 0.15
right = 0.99
wspace = 0.
hspace = 0.3
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)
leg.set_bbox_to_anchor( ( left, 0.005, right-left, right ) )
plt.savefig(out_folder+'/figure_4.pdf')
plt.close('all')
