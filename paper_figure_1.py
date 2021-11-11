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

# add path for correct version of tensiometer:
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
from tensiometer import utilities

###############################################################################
# initial settings:

import example_1_generate as example
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
y_size = 11.0
main_fontsize = 10.0

levels = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]
#param_ranges = [[0.0, 0.5], [0.3, 1.5]]
param_ranges = [[-2.2,-.6],[-.5,.4]]#np.log([[0.15, 0.5], [0.3, 1.5]])


# define the grid:
P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 200)
P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 200)
x, y = P1, P2
X, Y = np.meshgrid(x, y)

# compute probability:
log_P = example.posterior_flow.log_probability(np.array([X, Y], dtype=np.float32).T)
log_P = np.array(log_P).T
P = np.exp(log_P)
P = P / simps(simps(P, y), x)

# compute maximum posterior and metric:
#result = example.posterior_flow.MAP_finder(disp=True)
#maximum_posterior = result.x
maximum_posterior = example.posterior_flow.MAP_coord

# get fisher from samples:
fisher_metric = np.linalg.inv(example.posterior_chain.cov())

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(2, 1)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])

ax1.contour(X, Y, P, analyze_2d_example.get_levels(P, x, y, levels), linewidths=1., zorder=-1., linestyles='-', colors=[color_utilities.nice_colors(6) for i in levels])
ax2.contour(X, Y, P, analyze_2d_example.get_levels(P, x, y, levels), linewidths=1., zorder=-1., linestyles='-', colors=[color_utilities.nice_colors(6) for i in levels])

alpha = np.linspace(-1, 1, 1000)

# plot PCA of flow fisher metric modes:
eig, eigv = np.linalg.eigh(fisher_metric)
mode = 1
ax1.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1., color=color_utilities.nice_colors(1), ls='-')
mode = 0
ax1.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1., color=color_utilities.nice_colors(2), ls='-')

# correlation:
weights = np.diag(1./np.sqrt(np.diag(fisher_metric)))
fisher_correlation = np.dot(np.dot(weights, fisher_metric), weights)
eig, eigv = np.linalg.eigh(fisher_correlation)
eigv = np.dot(weights, eigv)
mode = 1
ax2.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1., color=color_utilities.nice_colors(1), ls='-')
mode = 0
ax2.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1., color=color_utilities.nice_colors(2), ls='-')


# Transform parameter basis:
A = np.array([[1, -1], [0, 1]])
fisher_metric_tilde = np.dot(np.dot(np.linalg.inv(A.T), fisher_metric), np.linalg.inv(A))

# plot PCA of transformed flow fisher metric modes:
eig, eigv = np.linalg.eigh(fisher_metric_tilde)
eigv = np.dot(A.T, eigv)
mode = 1
ax1.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1.5, color=color_utilities.nice_colors(0), ls=':')
mode = 0
ax1.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1.5, color=color_utilities.nice_colors(3), ls=':')

# correlation:
weights = np.diag(1./np.sqrt(np.diag(fisher_metric_tilde)))
fisher_correlation = np.dot(np.dot(weights, fisher_metric_tilde), weights)
eig, eigv = np.linalg.eigh(fisher_correlation)
eigv = np.dot(weights, eigv)
eigv = np.dot(A.T, eigv)
mode = 1
ax2.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1.5, color=color_utilities.nice_colors(0), ls=':')
mode = 0
ax2.axline(maximum_posterior, maximum_posterior+eig[mode]*eigv[:, mode], lw=1.5, color=color_utilities.nice_colors(3), ls=':')

# limits:
for ax in [ax1, ax2]:
    #ax.set_xlim([param_ranges[0][0], param_ranges[0][1]])
    #ax.set_ylim([-.6,.3])
    #ax.set_xlim([param_ranges[0][0], param_ranges[0][1]])
    #ax.set_ylim([0.4, 1.4])
    ax.set_xlim([-2.5, -0.5])
    ax.set_ylim([-0.6, 0.4])

# ticks:
ticks = [-2.5, -2.0, -1.5, -1.0, -0.5]
#ticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
for ax in [ax1, ax2]:
    ax.set_xticks(ticks)
ax1.set_xticklabels([], fontsize=0.9*main_fontsize);
ax2.set_xticklabels(ticks, fontsize=0.9*main_fontsize);
ax2.get_xticklabels()[0].set_horizontalalignment('left')
ax2.get_xticklabels()[-1].set_horizontalalignment('right')

ticks = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
#ticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
for ax in [ax1, ax2]:
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=0.9*main_fontsize);
    ax.get_yticklabels()[0].set_verticalalignment('bottom')
    ax.get_yticklabels()[-1].set_verticalalignment('top')

# axes labels:
ax2.set_xlabel(r'$\theta_1$', fontsize=main_fontsize);
for ax in [ax1, ax2]:
    ax.set_ylabel(r'$\theta_2$', fontsize=main_fontsize);

# title:
ax1.text( 0.01, 1.03, 'a) PCA of covariance matrix', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax1.transAxes)
ax2.text( 0.01, 1.03, 'b) PCA of correlation matrix', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax2.transAxes)

# legend:
from matplotlib.legend_handler import HandlerBase
class object_1():
    pass
class AnyObjectHandler1(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], color=color_utilities.nice_colors(1), lw=1.)
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color=color_utilities.nice_colors(2), lw=1.)
        return [l1, l2]

class object_2():
    pass
class AnyObjectHandler2(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], color=color_utilities.nice_colors(0), lw=1.5, ls=':')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color=color_utilities.nice_colors(3), lw=1.5, ls=':')
        return [l1, l2]

leg_handlers = [mlines.Line2D([], [], lw=1., ls='-', color='k'),
                object_1, object_2]
legend_labels = [r'$\mathcal{P}$', 'PC of $\\theta$', 'PC of $\\tilde{\\theta}$']

leg = fig.legend(handles=leg_handlers,
                labels=legend_labels,
                handler_map={object_1: AnyObjectHandler1(), object_2: AnyObjectHandler2()},
                fontsize=0.9*main_fontsize,
                frameon=True,
                fancybox=False,
                edgecolor='k',
                ncol=len(legend_labels),
                borderaxespad=0.0,
                columnspacing=2.0,
                handlelength=1.5,
                handletextpad=0.3,
                loc = 'lower center', #mode='expand',
                bbox_to_anchor=(0.0, 0.0, 0.9, 0.9),
                )
leg.get_frame().set_linewidth('0.8')

# update dimensions:
bottom = 0.16
top = 0.95
left = 0.15
right = 0.99
wspace = .2
hspace = 0.2
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)
leg.set_bbox_to_anchor((left, 0.005, right-left, right))

plt.savefig(out_folder+'/figure_1.pdf')
plt.close('all')
