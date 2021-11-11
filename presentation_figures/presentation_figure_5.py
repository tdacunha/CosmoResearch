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
from scipy.integrate import simps
from scipy import optimize

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
out_folder = './results/presentation_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# image size:
x_size = 8.54
y_size = 11.0
main_fontsize = 10.0

# color palette:
colors = [color_utilities.nice_colors(i) for i in range(6)]

# number of modes:
num_modes = 3

###############################################################################
# general utility functions:


# helper function to get levels:
def get_levels(P, x, y, conf=[0.95, 0.68]):
    """
    Get levels from a 2D grid
    """
    def _helper(alpha):
        _P = P.copy()
        _P[_P < alpha] = 0.
        return simps(simps(_P, y), x)
    levs = []
    for c in conf:
        try:
            res = optimize.brentq(lambda x: _helper(x)-c, np.amin(P), np.amax(P))
            levs.append(res)
        except:
            print('Cannot generate proper levels')
            levs = len(conf)
            break
    levs = np.sort(levs)
    return levs

###############################################################################
# get 2D flows:


log_param_names = ['log_omegam', 'log_sigma8']
log_params_flow_cache = out_folder+'log_params_flow_cache_3x2'
temp = DES_generate.helper_load_chains(log_param_names, example.prior_chain, example.posterior_chain, log_params_flow_cache, feedback=0)
log_params_prior_flow, log_params_posterior_flow = temp

param_names = ['omegam', 'sigma8']
params_flow_cache = out_folder+'params_flow_cache_3x2'
temp = DES_generate.helper_load_chains(param_names, example.prior_chain, example.posterior_chain, params_flow_cache)
params_prior_flow, params_posterior_flow = temp

###############################################################################
# do Local PCA:
num_params = len(log_param_names)

reference_point = example.posterior_chain.getMeans(pars=[example.posterior_chain.index[name] for name in log_param_names])

# local fisher:
fisher = log_params_posterior_flow.metric(log_params_posterior_flow.cast([reference_point]))[0]
prior_fisher = log_params_prior_flow.metric(log_params_prior_flow.cast([reference_point]))[0]

eig, eigv = utilities.KL_decomposition(fisher, prior_fisher)
sqrt_fisher = scipy.linalg.sqrtm(fisher)

# sort modes:
idx = np.argsort(eig)[::-1]
eig = eig[idx]
eigv = eigv[:, idx]

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# plot size in cm. Has to match to draft to make sure font sizes are consistent
x_size = 8.54
y_size = 6.5
main_fontsize = 10.0

levels = [utilities.from_sigma_to_confidence(i) for i in range(2, 0, -1)]

# obtain posterior on grid from samples:
density_1 = example.posterior_chain.get2DDensity(param_names[0], param_names[1], normalized=True)
_X1, _Y1 = np.meshgrid(density_1.x, density_1.y)
density_1.P = density_1.P / simps(simps(density_1.P, density_1.y), density_1.x)

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])

ax1.contour(_X1, _Y1, density_1.P, get_levels(density_1.P, density_1.x, density_1.y, levels), linewidths=1., linestyles='-', colors=[colors[0] for i in levels])

m1, m2 = np.exp(example.posterior_chain.getMeans([example.posterior_chain.index[name] for name in log_param_names]))
ax1.scatter(m1, m2, c=[colors[0]], edgecolors='white', zorder=999, s=20)

#ax1.plot(density_1.x, m2*(density_1.x/m1)**(-0.5), ls='--', lw=1., color='k')

# first mode:
temp = np.sqrt(eig[0])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
ax1.plot(m1*np.exp(alpha*eigv[0, 0]), m2*np.exp(alpha*eigv[1, 0]), c=colors[1], lw=1., ls='-')

# second mode:
temp = np.sqrt(eig[1])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
ax1.plot(m1*np.exp(alpha*eigv[0, 1]), m2*np.exp(alpha*eigv[1, 1]), c=colors[2], lw=1., ls='-')

levs = np.append(get_levels(density_1.P, density_1.x, density_1.y, levels), [np.amax(density_1.P)])
cols = [colors[0] for i in levs]
cols[0] = tuple(list(cols[0])+[0.1])
cols[1] = tuple(list(cols[1])+[0.8])
ax1.contourf(_X1, _Y1, density_1.P, levs, colors=cols)

# limits:
ax1.set_xlim([0.15, 0.4])
ax1.set_ylim([0.6, 1.1])

# ticks:
ticks = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_xticklabels()[0].set_horizontalalignment('left')
ax1.get_xticklabels()[-1].set_horizontalalignment('right')

ticks = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1]
ax1.set_yticks(ticks)
ax1.set_yticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_yticklabels()[0].set_verticalalignment('bottom')
ax1.get_yticklabels()[-1].set_verticalalignment('top')

# axes labels:
ax1.set_xlabel(r'Matter fractional density $\Omega_m$', fontsize=main_fontsize);
ax1.set_ylabel(r'Amplitude of fluctuations $\sigma_8$', fontsize=main_fontsize);

# legend:
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

leg_handlers = [
                mpatches.Patch(color=list(colors[0])+[0.8], ec=colors[0]),
                mlines.Line2D([], [], lw=1., ls='-', color=colors[1]),
                mlines.Line2D([], [], lw=1., ls='-', color=colors[2]),
                ]
legend_labels = [
                 'DES Y1 3x2',
                 'mode 1',
                 'mode 2'
                 ]

# legend for the second plot:
leg = fig.legend(handles=leg_handlers,
                   labels=legend_labels,
                   fontsize=0.8*main_fontsize,
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
bottom = 0.16
top = 0.99
left = 0.14
right = 0.99
wspace = 0.
hspace = 0.3
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)
leg.set_bbox_to_anchor((0.0, 0.0, right-0.01, top-0.01))

plt.savefig(out_folder+'/figure_5.pdf')
plt.close('all')
