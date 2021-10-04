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
colors = [color_utilities.nice_colors(i) for i in range(4)]

# number of modes:
num_modes = 2

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
# decide parameters to use:

pca_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'ns']
param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns']
num_params = len(param_names)

###############################################################################
# do PCA:

# compute covariance and PCA of fisher:
covariance = posterior_chain.cov(param_names)
fisher = np.linalg.inv(covariance)
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
        param1, param2 = param_names[i], param_names[i2+1]
        # create sub plot:
        g._subplot(i, i2, pars=(param1, param2),
                   sharex=g.subplots[bottom, i] if i2 != bottom else None,
                   sharey=g.subplots[i2, 0] if i > 0 else None)
        ax = g.subplots[i2, i]
        # add plot 2D:
        g.plot_2d([posterior_chain], param_pair=(param1, param2), do_xlabel=i2 == num_params - 2, do_ylabel=i == 0,
                  no_label_no_numbers=g.settings.no_triangle_axis_labels, shaded=False,
                  add_legend_proxy=i == 0 and i2 == 1, ax=ax, colors=colors, filled=True)
        g._inner_ticks(ax)
        # add PCA lines:
        m1, m2 = posterior_chain.getBestFit().parWithName(param1).best_fit, posterior_chain.getBestFit().parWithName(param2).best_fit


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
ax.text(0.01, 1.05, 'a) DES Y1 shear', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax.transAxes)




# update dimensions:
bottom = 0.12
top = 0.93
left = 0.12
right = 0.99
wspace = 0.
hspace = 0.
g.gridspec.update(bottom=bottom, top=top, left=left, right=right,
                  wspace=wspace, hspace=hspace)
#leg.set_bbox_to_anchor((0.0, 0.0, right, top))

# save:
g.fig.savefig(out_folder+'/figure_DES_LCDM_triangle.pdf')
