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
import color_utilities
import utilities as utils

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
# import the tensiometer tools that we need:
from tensiometer import utilities
# import example:
import example_DES_3x2 as example

###############################################################################
# initial settings:

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# image size:
x_size = 8.54
y_size = 11.0
main_fontsize = 10.0

# color palette:
colors = [color_utilities.nice_colors(i) for i in range(6)]

# number of modes:
num_modes = 3

###############################################################################
# do Local PCA:
#MAP  = example.log_params_posterior_flow.MAP_coord
means = []
for i in range(0, len(example.log_param_names)):
    param_i = example.log_param_names[i]
    mean_i = example.posterior_chain.getMeans([example.posterior_chain.index[param_i]])[0]
    means.append(mean_i)
MAP = means
#MAP = example.log_params_posterior_flow.sample_MAP


# compute PCA of local fisher:
num_params = len(example.log_param_names)
covariance = example.posterior_chain.cov(example.log_param_names)
fisher = example.log_params_posterior_flow.metric(example.log_params_posterior_flow.cast([MAP]))[0]
eig, eigv = np.linalg.eigh(fisher)
sqrt_fisher = scipy.linalg.sqrtm(fisher)
# sort modes:
idx = np.argsort(eig)[::-1]
eig = eig[idx]
eigv = eigv[:, idx]

###############################################################################
# plot:

# create figure:
g = plots.getSubplotPlotter(width_inch=x_size/2.54)
g.settings.num_plot_contours = 2
g.settings.solid_contour_palefactor = 0.6
g.settings.alpha_factor_contour_lines = 1.0
g.settings.fontsize = main_fontsize
g.settings.axes_fontsize = 0.9*main_fontsize
g.settings.lab_fontsize = main_fontsize
g.settings.legend_fontsize = 0.9*main_fontsize
g.settings.figure_legend_loc = 'upper right'
g.settings.figure_legend_ncol = 1
g.settings.legend_frame = True
g.settings.axis_marker_lw = 1.
g.settings.x_label_rotation = 0.
g.settings.lw_contour = 1.

g.make_figure(nx=num_params-1, ny=num_params-1, sharex=g.settings.no_triangle_axis_labels,
              sharey=g.settings.no_triangle_axis_labels)

# create subplots:
bottom = num_params - 2
for i in range(num_params-1):
    for i2 in range(bottom, i-1, -1):
        param1, param2 = example.param_names[i], example.param_names[i2+1]
        # create sub plot:
        g._subplot(i, i2, pars=(param1, param2),
                   sharex=g.subplots[bottom, i] if i2 != bottom else None,
                   sharey=g.subplots[i2, 0] if i > 0 else None)
        ax = g.subplots[i2, i]
        # add plot 2D:
        g.plot_2d([example.posterior_chain], param_pair=(param1, param2), do_xlabel=i2 == num_params - 2, do_ylabel=i == 0,
                  no_label_no_numbers=g.settings.no_triangle_axis_labels, shaded=False,
                  add_legend_proxy=i == 0 and i2 == 1, ax=ax, colors=colors, filled=True)
        g._inner_ticks(ax)
        # add PCA lines:
        #m1, m2 = example.posterior_chain.getBestFit().parWithName(param1).best_fit, example.posterior_chain.getBestFit().parWithName(param2).best_fit
        #m1 = example.posterior_chain.getMeans([example.posterior_chain.index[param1]])[0]
        #m2 = example.posterior_chain.getMeans([example.posterior_chain.index[param2]])[0]
        #map1 = np.exp(example.log_params_posterior_flow.MAP_coord[i])
        #map2 = np.exp(example.log_params_posterior_flow.MAP_coord[i2+1])
        map1,map2 = np.exp(MAP[i]),np.exp(MAP[i2+1])
        #ax.scatter([m1], [m2], c=[colors[0]], edgecolors='black', zorder=999, s=20)
        ax.scatter(map1, map2, c=[colors[0]], edgecolors='white', zorder=999, s=20)

        for k in range(num_modes):
            idx1 = example.param_names.index(param1)
            idx2 = example.param_names.index(param2)
            temp = np.sqrt(eig[k])
            alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
            ax.plot(map1*np.exp(alpha*eigv[idx1, k]), map2*np.exp(alpha*eigv[idx2, k]), c=colors[k+1], lw=1., ls='-', zorder=998, label='PC mode '+str(k+1))
            #ax.plot(m1 + alpha*eigv[idx1, k], m2 + alpha*eigv[idx2, k], c=colors[k+1], lw=1., ls='-', zorder=998, label='PC mode '+str(k+1))

# ticks:
for _row in g.subplots:
    for _ax in _row:
        if _ax is not None:
            _ax.tick_params('both', length=2.5, width=.8,
                            which='major', zorder=999,
                            labelsize=0.9*main_fontsize)
            _ax.xaxis.label.set_size(main_fontsize)
            _ax.yaxis.label.set_size(main_fontsize)

# update the settings:
g.fig.set_size_inches(x_size/2.54, y_size/2.54)

# text:
ax = g.subplots[0, 0]
ax.text(0.01, 1.05, 'b) DES Y1 3x2', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax.transAxes)

# legend:
leg_handlers, legend_labels = ax.get_legend_handles_labels()

# legend for the second plot:
leg = g.fig.legend(handles=leg_handlers,
                   labels=legend_labels,
                   fontsize=0.9*main_fontsize,
                   frameon=True,
                   fancybox=False,
                   edgecolor='k',
                   ncol=1,
                   borderaxespad=0.0,
                   columnspacing=2.0,
                   handlelength=1.4,
                   loc='upper right',
                   bbox_to_anchor=(0.0, 0.0, 0.9, 0.9),
                   )
leg.get_frame().set_linewidth('0.8')
leg.get_title().set_fontsize(main_fontsize)

# update dimensions:
bottom = 0.10
top = 0.93
left = 0.15
right = 0.99
wspace = 0.
hspace = 0.
g.gridspec.update(bottom=bottom, top=top, left=left, right=right,
                  wspace=wspace, hspace=hspace)
leg.set_bbox_to_anchor((0.0, 0.0, right, top))

# save:
g.fig.savefig(out_folder+'/figure_DES_LCDM_triangle_2.pdf')
