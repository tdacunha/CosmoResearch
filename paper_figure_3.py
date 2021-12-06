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
import matplotlib.cm as cm


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
y_size = 11.0
main_fontsize = 10.0

levels = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]
#param_ranges = [[0.0, 0.5], [0.3, 1.5]]
#param_ranges = [[-2.2,-.6],[-.5,.4]]#np.log([[0.15, 0.5], [0.3, 1.5]])
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

# original space:
ax1.contour(X, Y, P, analyze_2d_example.get_levels(P, x, y, levels), linewidths=1., zorder=-1., linestyles='-', colors=[color_utilities.nice_colors(6) for i in levels])
# MAP:
ax1.scatter(*maximum_posterior, s=5.0, color='k', zorder=999)

# abstract space:
theta = np.linspace(0.0, 2.*np.pi, 200)
for i in range(4):
    _length = np.sqrt(stats.chi2.isf(1.-utilities.from_sigma_to_confidence(i), 2))
    ax2.plot(_length*np.sin(theta), _length*np.cos(theta), ls='-', zorder=999., lw=1., color='k')

# MAP:
reference_image = example.posterior_flow.map_to_abstract_coord(example.posterior_flow.cast(maximum_posterior))
ax2.scatter(*reference_image, s=5.0, color='k', zorder=999)
# compute geodesics at range of angles:
reference_image = example.posterior_flow.map_to_abstract_coord(example.posterior_flow.cast(maximum_posterior))
r = np.linspace(0.0, example.posterior_flow.sigma_to_length(3.), 1000)
theta = np.linspace(0.0, 2.0*np.pi, 20)
geodesics, abs_geodesics = [], []
for t in theta:
    geo = np.array([reference_image[0] + r*np.cos(t), reference_image[1] + r*np.sin(t)], dtype=np.float32)
    geodesics.append(example.posterior_flow.map_to_original_coord(geo.T))
    abs_geodesics.append(geo.T)

cmap = cm.get_cmap('Spectral')
for ind, geo in enumerate(geodesics):
    ax1.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)

for ind, geo in enumerate(abs_geodesics):
    ax2.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)

# limits:
ax1.set_xlim([0.1, 0.45])
ax1.set_ylim([0.6, 1.2])

ax2.set_xlim([-3.0, 3.0])
ax2.set_ylim([-4.0, 4.0])

# ticks:
ticks1 = [0.1, 0.2, 0.3, 0.4, 0.45]
ticks2 = [-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]

ax1.set_xticks(ticks1)
ax2.set_xticks(ticks2)
ax1.set_xticklabels([], fontsize=0.9*main_fontsize);
ax1.set_xticklabels(ticks1, fontsize=0.9*main_fontsize);
ax2.set_xticklabels(ticks2, fontsize=0.9*main_fontsize);

ax1.get_xticklabels()[0].set_horizontalalignment('left')
ax1.get_xticklabels()[-1].set_horizontalalignment('right')

ax2.get_xticklabels()[0].set_horizontalalignment('left')
ax2.get_xticklabels()[-1].set_horizontalalignment('right')

ticks1 = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
ticks2 = [-4, -3, -2, -1, 0, 1, 2, 3, 4]

#ticks = [0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
ax1.set_yticks(ticks1)
ax1.set_yticklabels(ticks1, fontsize=0.9*main_fontsize);
ax1.get_yticklabels()[0].set_verticalalignment('bottom')
ax1.get_yticklabels()[-1].set_verticalalignment('top')
ax2.set_yticks(ticks2)
ax2.set_yticklabels(ticks2, fontsize=0.9*main_fontsize);
ax2.get_yticklabels()[0].set_verticalalignment('bottom')
ax2.get_yticklabels()[-1].set_verticalalignment('top')

# axes labels:
ax1.set_xlabel(r'$\theta_1$', fontsize=main_fontsize);
ax1.set_ylabel(r'$\theta_2$', fontsize=main_fontsize);
ax2.set_xlabel(r'$Z_1$', fontsize=main_fontsize);
ax2.set_ylabel(r'$Z_2$', fontsize=main_fontsize);

# title:
ax1.text( 0.01, 1.03, 'a) Original parameter space', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax1.transAxes)
ax2.text( 0.01, 1.03, 'b) Abstract parameter space', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax2.transAxes)

# legend:
from matplotlib.legend_handler import HandlerBase
class object_1():
    pass
class AnyObjectHandler1(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.75*height,0.75*height], color=cmap(0/len(geodesics)), lw=1.)
        l2 = plt.Line2D([x0,y0+width], [0.45*height,0.45*height], color=cmap(6/len(geodesics)), lw=1.)
        l3 = plt.Line2D([x0,y0+width], [0.15*height,0.15*height], color=cmap(17/len(geodesics)), lw=1.)

        return [l1, l2, l3]

# class object_2():
#     pass
# class AnyObjectHandler2(HandlerBase):
#     def create_artists(self, legend, orig_handle,
#                        x0, y0, width, height, fontsize, trans):
#         l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], color=color_utilities.nice_colors(0), lw=1.5, ls=':')
#         l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color=color_utilities.nice_colors(3), lw=1.5, ls=':')
#         return [l1, l2]

leg_handlers = [mlines.Line2D([], [], lw=1., ls='-', color='k'),
                object_1]
legend_labels = [r'$\mathcal{P}$', 'Geodesics']

leg = fig.legend(handles=leg_handlers,
                labels=legend_labels,
                handler_map={object_1: AnyObjectHandler1()},
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
hspace = 0.4
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)
leg.set_bbox_to_anchor((left, 0.005, right-left, right))

plt.savefig(out_folder+'/figure_3.pdf')
plt.close('all')
