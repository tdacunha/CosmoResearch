# -*- coding: utf-8 -*-

###############################################################################
# initial imports:

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.integrate import simps
from getdist import plots, MCSamples
import color_utilities
import getdist
getdist.chains.print_load_details = False
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# add path for correct version of tensiometer:
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
from tensiometer import utilities

###############################################################################
# initial settings:

import example_3_generate as example
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

levels = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]
param_ranges = [[-0.999, 0.999], [-0.999, 0.999]]

# define the grid:
P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 300)
P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 300)
x, y = P1, P2
X, Y = np.meshgrid(x, y)

# compute probability:
log_P = example.posterior_flow.log_probability(np.array([X, Y], dtype=np.float32).T)
log_P = np.array(log_P).T
P = np.exp(log_P)
P = P / simps(simps(P, y), x)

# compute maximum posterior:
result = example.posterior_flow.MAP_finder(disp=True)
maximum_posterior = result.x

# get the mean:
mean = example.posterior_chain.getMeans([example.posterior_chain.index[name] for name in example.posterior_chain.getParamNames().list()])

# compute the two base eigenvalues trajectories:

# compute eigenvalue network:


# plot size in cm. Has to match to draft to make sure font sizes are consistent
x_size = 18.0
y_size = 7.5
main_fontsize = 10.0

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

# plot the pdf:
ax1.contour(X, Y, P, analyze_2d_example.get_levels(P, x, y, levels), linewidths=1., zorder=-1., linestyles='-', colors=[color_utilities.nice_colors(6) for i in levels])

theta = np.linspace(0.0, 2.*np.pi, 200)
for i in range(4):
    _length = np.sqrt(stats.chi2.isf(1.-utilities.from_sigma_to_confidence(i), 2))
    ax2.plot(_length*np.sin(theta), _length*np.cos(theta), ls='-', zorder=-1., lw=1., color='k')

# MAP:
ax1.scatter(*maximum_posterior, s=5.0, color='k')
ax2.scatter(*example.posterior_flow.map_to_abstract_coord(example.posterior_flow.cast(maximum_posterior)), s=5.0, color='k')

# mean:
ax1.scatter(*mean, s=20.0, marker='x', color='k')
ax2.scatter(*example.posterior_flow.map_to_abstract_coord(example.posterior_flow.cast(mean)), s=20.0, marker='x', color='k')

# prior:
ax1.axvspan(1., 1.2, alpha=0.2, ec=None, color='k')
ax1.axvspan(-1., -1.2, alpha=0.2, ec=None, color='k')
ax1.fill_between([-1., 1.], [1., 1.], [1.2, 1.2], alpha=0.2, ec=None, lw=0.0, color='k')
ax1.fill_between([-1., 1.], [-1., -1.], [-1.2, -1.2], alpha=0.2, ec=None, lw=0.0, color='k')
ax1.add_patch(Rectangle((-1., -1.), 2.0, 2.0, fill=None, alpha=1, color='k', ls='--', lw=1.))

# limits:
ax1.set_xlim([-1.1, 1.1])
ax1.set_ylim([-1.1, 1.1])

ax2.set_xlim([-4.0, 4.0])
ax2.set_ylim([-4.0, 4.0])

## ticks:
#ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#for ax in [ax1, ax2]:
#    ax.set_xticks(ticks)
#ax1.set_xticklabels([], fontsize=0.9*main_fontsize);
#ax2.set_xticklabels(ticks, fontsize=0.9*main_fontsize);
#ax2.get_xticklabels()[0].set_horizontalalignment('left')
#ax2.get_xticklabels()[-1].set_horizontalalignment('right')

#ticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
#for ax in [ax1, ax2]:
#    ax.set_yticks(ticks)
#    ax.set_yticklabels(ticks, fontsize=0.9*main_fontsize);
#    ax.get_yticklabels()[0].set_verticalalignment('bottom')
#    ax.get_yticklabels()[-1].set_verticalalignment('top')

# axes labels:
ax1.set_xlabel(r'$\theta_1$', fontsize=main_fontsize);
ax1.set_ylabel(r'$\theta_2$', fontsize=main_fontsize);
ax2.set_xlabel(r'$Z_1$', fontsize=main_fontsize);
ax2.set_ylabel(r'$Z_2$', fontsize=main_fontsize);

## title:
#ax1.text( 0.01, 1.03, 'a) PCA of covariance matrix', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax1.transAxes)
#ax2.text( 0.01, 1.03, 'b) PCA of correlation matrix', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax2.transAxes)

# update dimensions:
bottom = 0.16
top = 0.95
left = 0.15
right = 0.99
wspace = 0.2
hspace = 0.2
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)

plt.savefig(out_folder+'/figure_example_3.pdf')
plt.close('all')
