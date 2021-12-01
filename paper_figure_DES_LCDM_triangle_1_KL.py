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

# image size:
x_size = 8.54
y_size = 11.0
main_fontsize = 10.0

# color palette:
colors = [color_utilities.nice_colors(i) for i in range(6)]
line_colors = [1,0,2]
# number of modes:
num_modes = 2

###############################################################################
# do local KL:

num_params = len(example.lcdm_shear_params_log_param_names)
# reference point:
reference_point = np.log([name.best_fit for name in example.posterior_chain_lcdm_shear.getBestFit().parsWithNames([name.replace('log_', '') for name in example.lcdm_shear_params_log_param_names])])
reference_point = np.array([example.posterior_chain_lcdm_shear.samples[np.argmin(example.posterior_chain_lcdm_shear.loglikes), :][example.posterior_chain_lcdm_shear.index[name]] for name in example.lcdm_shear_params_log_param_names])
reference_point = example.posterior_chain_lcdm_shear.getMeans(pars=[example.posterior_chain_lcdm_shear.index[name] for name in example.lcdm_shear_params_log_param_names])
# local fisher:
fisher = example.lcdm_shear_log_params_posterior_flow.metric(example.lcdm_shear_log_params_posterior_flow.cast([reference_point]))[0]
prior_fisher = example.lcdm_shear_log_params_prior_flow.metric(example.lcdm_shear_log_params_prior_flow.cast([reference_point]))[0]
# global fisher:
#fisher = np.linalg.inv(example.posterior_chain_lcdm_shear.cov(example.lcdm_shear_params_log_param_names))
#prior_fisher = np.linalg.inv(example.prior_chain_lcdm_shear.cov(example.lcdm_shear_params_log_param_names))

eig, eigv = utilities.KL_decomposition(fisher, prior_fisher)
sqrt_fisher = scipy.linalg.sqrtm(fisher)

# sort modes:
idx = np.argsort(eig)[::-1]
eig = eig[idx]
eigv = eigv[:, idx]

# print out modes:
temp = np.dot(sqrt_fisher, eigv)
contributions = temp * temp / eig
for i in range(num_modes):
    idx_max = np.argmax(contributions[:, i])
    print('* Mode', i+1)
    print('  Sqrt eig = ', np.round(np.sqrt(eig[i]), 2))
    _directions = np.linalg.inv(eigv).T
    _norm_eigv = _directions[:, i] / _directions[idx_max, i]
    # normalize to fixed param:
    ref_idx = example.lcdm_shear_params_log_param_names.index('log_sigma8')
    _norm_eigv = _directions[:, i] / _directions[ref_idx, i]
    with np.printoptions(precision=2, suppress=True):
        print('  Variance contributions', contributions[:, i])
    string = ''
    for j in range(num_params):
        _name = example.lcdm_shear_params_log_param_names[j]
        _mean = reference_point[j]
        _mean = '{0:+}'.format(np.round(-_mean, 2))
        _temp = '{0:+}'.format(np.round(_norm_eigv[j], 2))
        string += _temp+'*('+_name+' '+_mean+') '
    print('  Mode =', string, '= 0')
    print(' ')

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
        param1, param2 = example.lcdm_shear_params_param_names[i], example.lcdm_shear_params_param_names[i2+1]
        # create sub plot:
        g._subplot(i, i2, pars=(param1, param2),
                   sharex=g.subplots[bottom, i] if i2 != bottom else None,
                   sharey=g.subplots[i2, 0] if i > 0 else None)
        ax = g.subplots[i2, i]
        # add plot 2D:
        g.plot_2d([example.posterior_chain_lcdm_shear], param_pair=(param1, param2), do_xlabel=i2 == num_params - 2, do_ylabel=i == 0,
                  no_label_no_numbers=g.settings.no_triangle_axis_labels, shaded=False,
                  add_legend_proxy=i == 0 and i2 == 1, ax=ax, colors=['lightslategrey'], filled=True)
        g._inner_ticks(ax)
        # add PCA lines:
        m1, m2 = np.exp(reference_point[i]), np.exp(reference_point[i2+1])
        ax.scatter(m1, m2, c=[colors[3]], edgecolors='white', zorder=999, s=20)

        for k in range(num_modes):
            idx1 = example.lcdm_shear_params_param_names.index(param1)
            idx2 = example.lcdm_shear_params_param_names.index(param2)
            temp = np.sqrt(eig[k])
            alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
            ax.plot(m1*np.exp(alpha*eigv[idx1, k]), m2*np.exp(alpha*eigv[idx2, k]), c=colors[line_colors[k]], lw=1., ls='-', zorder=998, label='CPC mode '+str(k+1))

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
g.fig.savefig(out_folder+'/figure_DES_LCDM_triangle_1_KL.pdf')
