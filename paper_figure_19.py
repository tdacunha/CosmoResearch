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
import synthetic_probability
from tensorflow_probability import bijectors as bj

# import example:
import example_DES_Y1 as example

###############################################################################
# initial settings:

do_linear = False

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# image size:
x_size = 8.54
y_size = 11.0
main_fontsize = 10.0

# color palette:
colors = [color_utilities.nice_colors(i) for i in range(6)]
line_colors = [1, 0, 2]
# number of modes:
num_modes = 2

###############################################################################
# get the flows:

# param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns', 'DES_AIA', 'DES_alphaIA']
# log_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'log_ns', 'DES_AIA', 'DES_alphaIA']
# transformation = [bj.Log(), bj.Log(), bj.Log(), bj.Log(), bj.Log(), example.identity_bj(), example.identity_bj()]
# num_params = len(param_names)

param_names = ['omegam', 'sigma8',  'omegab', 'H0', 'DES_AIA']
log_param_names = ['log_omegam', 'log_sigma8', 'log_omegab', 'log_H0', 'DES_AIA']
transformation = [bj.Log(), bj.Log(), bj.Log(),  bj.Log(), example.identity_bj()]
num_params = len(param_names)

# with boundaries:
params_flow_cache = './results/example_DES_Y1_bound/'+'lcdm_shear_params_full_flow_cache_plot'
temp = DES_generate.helper_load_chains(param_names, example.prior_chain_lcdm_shear, example.posterior_chain_lcdm_shear, params_flow_cache)
shear_params_prior_flow_bound, shear_params_posterior_flow_bound = temp
shear_log_params_prior_flow_bound = synthetic_probability.TransformedDiffFlowCallback(temp[0], transformation)
shear_log_params_posterior_flow_bound = synthetic_probability.TransformedDiffFlowCallback(temp[1], transformation)

params_flow_cache = './results/example_DES_Y1_bound/'+'lcdm_3x2_params_shear_flow_cache_plot'
temp = DES_generate.helper_load_chains(param_names, example.prior_chain_lcdm_3x2, example.posterior_chain_lcdm_3x2, params_flow_cache)
des3x2_params_prior_flow_bound, des3x2_params_posterior_flow_bound = temp
des3x2_log_params_prior_flow_bound = synthetic_probability.TransformedDiffFlowCallback(temp[0], transformation)
des3x2_log_params_posterior_flow_bound = synthetic_probability.TransformedDiffFlowCallback(temp[1], transformation)

# without boundaries:
params_flow_cache = './results/example_DES_Y1_nobound/'+'lcdm_shear_params_full_flow_cache_plot'
temp = DES_generate.helper_load_chains(param_names, example.prior_chain_lcdm_shear, example.posterior_chain_lcdm_shear, params_flow_cache)
shear_params_prior_flow_nobound, shear_params_posterior_flow_nobound = temp
shear_log_params_prior_flow_nobound = synthetic_probability.TransformedDiffFlowCallback(temp[0], transformation)
shear_log_params_posterior_flow_nobound = synthetic_probability.TransformedDiffFlowCallback(temp[1], transformation)

params_flow_cache = './results/example_DES_Y1_nobound/'+'lcdm_3x2_params_shear_flow_cache_plot'
temp = DES_generate.helper_load_chains(param_names, example.prior_chain_lcdm_3x2, example.posterior_chain_lcdm_3x2, params_flow_cache)
des3x2_params_prior_flow_nobound, des3x2_params_posterior_flow_nobound = temp
des3x2_log_params_prior_flow_nobound = synthetic_probability.TransformedDiffFlowCallback(temp[0], transformation)
des3x2_log_params_posterior_flow_nobound = synthetic_probability.TransformedDiffFlowCallback(temp[1], transformation)

###############################################################################
# do linear CPCA:

# get starting point:
reference_point_3x2 = example.posterior_chain_lcdm_3x2.getMeans(pars=[example.posterior_chain_lcdm_3x2.index[name] for name in log_param_names])
reference_point_shear = example.posterior_chain_lcdm_shear.getMeans(pars=[example.posterior_chain_lcdm_shear.index[name] for name in log_param_names])
# local fisher:
fisher = des3x2_log_params_posterior_flow_bound.metric(des3x2_log_params_posterior_flow_bound.cast([reference_point_3x2]))[0]
prior_fisher = shear_log_params_posterior_flow_bound.metric(shear_log_params_posterior_flow_bound.cast([reference_point_3x2]))[0]
# get modes:
eig, eigv = utilities.KL_decomposition(fisher, prior_fisher)
idx = np.argsort(eig)[::-1]
eig = eig[idx]
eigv = eigv[:, idx]

###############################################################################
# do non-linear CPCA:

if do_linear:
    # get starting point:
    reference_point_3x2_2 = [trans.inverse(p).numpy() for p, trans in zip(reference_point_3x2, transformation)]
    reference_point_shear_2 = [trans.inverse(p).numpy() for p, trans in zip(reference_point_shear, transformation)]
    # solve for modes:
    y0 = des3x2_params_posterior_flow_nobound.cast(reference_point_3x2_2)
    length_1 = 20
    length_2 = 20
    _, LKL_mode_1, _ = synthetic_probability.solve_KL_ode(des3x2_params_posterior_flow_nobound, shear_params_posterior_flow_nobound, y0, n=-1, length=length_1, num_points=100)
    _, LKL_mode_2, _ = synthetic_probability.solve_KL_ode(des3x2_params_posterior_flow_nobound, shear_params_posterior_flow_nobound, y0, n=-2, length=length_2, num_points=100)
else:
    # get starting point:
    reference_point_3x2_2 = [trans.inverse(p).numpy() for p, trans in zip(reference_point_3x2, transformation)]
    reference_point_shear_2 = [trans.inverse(p).numpy() for p, trans in zip(reference_point_shear, transformation)]
    # solve for modes:
    y0 = des3x2_params_posterior_flow_nobound.cast(reference_point_3x2)
    length_1 = 20
    length_2 = 20
    _, LKL_mode_1, _ = synthetic_probability.solve_KL_ode(des3x2_log_params_posterior_flow_nobound, shear_log_params_posterior_flow_nobound, y0, n=-1, length=length_1, num_points=1000)
    _, LKL_mode_2, _ = synthetic_probability.solve_KL_ode(des3x2_log_params_posterior_flow_nobound, shear_log_params_posterior_flow_nobound, y0, n=-2, length=length_2, num_points=1000)
    # transform:
    for ind in range(num_params):
        LKL_mode_1[:, ind] = transformation[ind].inverse(LKL_mode_1[:, ind]).numpy()
        LKL_mode_2[:, ind] = transformation[ind].inverse(LKL_mode_2[:, ind]).numpy()

###############################################################################
# plot:

params_to_use = ['omegam', 'sigma8', 'omegab', 'H0', 'DES_AIA']
num_params_to_use = len(params_to_use)

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

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

g.make_figure(nx=num_params_to_use-1, ny=num_params_to_use-1, sharex=g.settings.no_triangle_axis_labels,
              sharey=g.settings.no_triangle_axis_labels)

# change label:
example.posterior_chain_lcdm_shear.getParamNames().parWithName('DES_AIA').label = 'A_{\\rm IA}'

# create subplots:
bottom = num_params_to_use - 2
for i in range(num_params_to_use-1):
    for i2 in range(bottom, i-1, -1):
        param1, param2 = params_to_use[i], params_to_use[i2+1]
        # create sub plot:
        g._subplot(i, i2, pars=(param1, param2),
                   sharex=g.subplots[bottom, i] if i2 != bottom else None,
                   sharey=g.subplots[i2, 0] if i > 0 else None)
        ax = g.subplots[i2, i]
        # add plot 2D:
        g.plot_2d([example.posterior_chain_lcdm_shear, example.posterior_chain_lcdm_3x2], param_pair=(param1, param2), do_xlabel=i2 == num_params_to_use - 2, do_ylabel=i == 0,
                  no_label_no_numbers=g.settings.no_triangle_axis_labels, shaded=False,
                  add_legend_proxy=i == 0 and i2 == 0, ax=ax, colors=[colors[2], colors[3]], filled=True)
        g._inner_ticks(ax)
        # indexes:
        idx1 = param_names.index(param1)
        idx2 = param_names.index(param2)

        # add PCA lines:
        ax.scatter(reference_point_3x2_2[idx1], reference_point_3x2_2[idx2], c=[colors[3]], edgecolors='white', zorder=999, s=20)
        ax.scatter(reference_point_shear_2[idx1], reference_point_shear_2[idx2], c=[colors[2]], edgecolors='white', zorder=999, s=20)

        m1, m2 = reference_point_3x2[idx1], reference_point_3x2[idx2]

        for k in range(num_modes):
            temp = np.sqrt(eig[k])
            alpha = 20.*np.linspace(-1./temp, 1./temp, 1000)
            ax.plot(transformation[idx1].inverse(m1+alpha*eigv[idx1, k]).numpy(),
                    transformation[idx2].inverse(m2+alpha*eigv[idx2, k]).numpy(),
                    c=colors[line_colors[k]], lw=1., ls='--', zorder=998, label='lin CPCC mode '+str(k+1))

        ax.plot(LKL_mode_1[:, idx1], LKL_mode_1[:, idx2], c=colors[line_colors[0]], lw=1., ls='-', zorder=998, label='non-lin CPCC mode 1')
        ax.plot(LKL_mode_2[:, idx1], LKL_mode_2[:, idx2], c=colors[line_colors[1]], lw=1., ls='-', zorder=998, label='non-lin CPCC mode 2')

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
ax.text(0.01, 1.05, 'DES Y1 3x2 improvement over shear', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax.transAxes)

# legend:
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

from matplotlib.legend_handler import HandlerBase
class object_1():
    pass
class AnyObjectHandler1(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], color=color_utilities.nice_colors(1), lw=1.2, ls = '--')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color=color_utilities.nice_colors(0), lw=1.2, ls = '--')
        return [l1, l2]

class object_2():
    pass
class AnyObjectHandler2(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], color=color_utilities.nice_colors(1), lw=1.2, ls='-')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color=color_utilities.nice_colors(0), lw=1.2, ls='-')
        return [l1, l2]

leg_handlers = [mpatches.Patch(color=list(colors[2])+[0.8], ec=colors[2]),
                mpatches.Patch(color=list(colors[3])+[0.8], ec=colors[3]),
                object_1, object_2]

legend_labels = [
                 'DES shear',
                 'DES 3x2',
                 'Lin. CPCC',
                 'Non-lin. CPCC',
                 ]

# legend for the second plot:
leg = g.fig.legend(handles=leg_handlers,
                   labels=legend_labels,
                   handler_map={object_1: AnyObjectHandler1(), object_2: AnyObjectHandler2()},
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
top = 0.94
left = 0.15
right = 0.99
wspace = 0.
hspace = 0.
g.gridspec.update(bottom=bottom, top=top, left=left, right=right,
                  wspace=wspace, hspace=hspace)
leg.set_bbox_to_anchor((0.0, 0.0, right, top))
# save:
g.fig.savefig(out_folder+'/figure_19.pdf')
