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
import synthetic_probability

###############################################################################
# initial settings:

import example_DES_Y3 as example

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# color palette:
colors = [color_utilities.nice_colors(i) for i in range(6)]

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


##############################################################################
# compute non linear modes:

# LCDM:
reference_point = example.posterior_chain_lcdm_shear.getMeans([example.posterior_chain_lcdm_shear.index[name] for name in example.lcdm_shear_log_params_posterior_flow.param_names])
y0 = example.lcdm_shear_log_params_posterior_flow.cast(reference_point)
length_1 = 2
length_2 = 10
_, lcdm_LKL_mode_1, _ = synthetic_probability.solve_KL_ode(example.lcdm_shear_log_params_posterior_flow, example.lcdm_shear_log_params_prior_flow, y0, n=-1, length=length_1, num_points=1000,
                                                           integrator_options={'name': 'lsoda', 'atol': 1.e-5, 'rtol': 1.e-5})
_, lcdm_LKL_mode_2, _ = synthetic_probability.solve_KL_ode(example.lcdm_shear_log_params_posterior_flow, example.lcdm_shear_log_params_prior_flow, y0, n=-2, length=length_2, num_points=1000,
                                                           integrator_options={'name': 'lsoda', 'atol': 1.e-5, 'rtol': 1.e-5})



plt.plot(np.exp(lcdm_LKL_mode_1[:, example.lcdm_shear_log_params_posterior_flow.param_names.index('log_omegam')]),
         np.exp(lcdm_LKL_mode_1[:, example.lcdm_shear_log_params_posterior_flow.param_names.index('log_sigma8')]),
         c=colors[1], lw=1., ls='-')
plt.plot(np.exp(lcdm_LKL_mode_2[:, example.lcdm_shear_log_params_posterior_flow.param_names.index('log_omegam')]),
         np.exp(lcdm_LKL_mode_2[:, example.lcdm_shear_log_params_posterior_flow.param_names.index('log_sigma8')]),
         c=colors[2], lw=1., ls='-')






# local fisher:
reference_point = example.posterior_chain_lcdm_shear.getMeans([example.posterior_chain_lcdm_shear.index[name] for name in example.lcdm_shear_log_params_posterior_flow.param_names])
lcdm_fisher = example.lcdm_shear_log_params_posterior_flow.metric(example.lcdm_shear_log_params_posterior_flow.cast([reference_point]))[0]
lcdm_prior_fisher = example.lcdm_shear_log_params_prior_flow.metric(example.lcdm_shear_log_params_prior_flow.cast([reference_point]))[0]
lcdm_eig, lcdm_eigv = utilities.KL_decomposition(lcdm_fisher, lcdm_prior_fisher)
# sort modes:
idx = np.argsort(lcdm_eig)[::-1]
lcdm_eig = lcdm_eig[idx]
lcdm_eigv = lcdm_eigv[:, idx]

#import tensorflow as tf
#t = 0
#flow = example.lcdm_shear_log_params_posterior_flow
#prior_flow = example.lcdm_shear_log_params_prior_flow
#y = flow.cast(reference_point)
#reference = eigv[:, 0]

# mnu:
reference_point = example.posterior_chain_mnu_shear.getMeans([example.posterior_chain_mnu_shear.index[name] for name in example.mnu_shear_log_params_posterior_flow.param_names])
y0 = example.mnu_shear_log_params_posterior_flow.cast(reference_point)
length_1 = 10
length_2 = 15
_, mnu_LKL_mode_1, _ = synthetic_probability.solve_KL_ode(example.mnu_shear_log_params_posterior_flow, example.mnu_shear_log_params_prior_flow, y0, n=-1, length=length_1, num_points=1000)
_, mnu_LKL_mode_2, _ = synthetic_probability.solve_KL_ode(example.mnu_shear_log_params_posterior_flow, example.mnu_shear_log_params_prior_flow, y0, n=-2, length=length_2, num_points=1000)

# local fisher:
reference_point = example.posterior_chain_mnu_shear.getMeans([example.posterior_chain_mnu_shear.index[name] for name in example.mnu_shear_log_params_posterior_flow.param_names])
mnu_fisher = example.mnu_shear_log_params_posterior_flow.metric(example.mnu_shear_log_params_posterior_flow.cast([reference_point]))[0]
mnu_prior_fisher = example.mnu_shear_log_params_prior_flow.metric(example.mnu_shear_log_params_prior_flow.cast([reference_point]))[0]
mnu_eig, mnu_eigv = utilities.KL_decomposition(mnu_fisher, mnu_prior_fisher)
# sort modes:
idx = np.argsort(mnu_eig)[::-1]
mnu_eig = mnu_eig[idx]
mnu_eigv = mnu_eigv[:, idx]

# wCDM:
reference_point = example.posterior_chain_wcdm_shear.getMeans([example.posterior_chain_wcdm_shear.index[name] for name in example.wcdm_shear_log_params_posterior_flow.param_names])
y0 = example.wcdm_shear_log_params_posterior_flow.cast(reference_point)
length_1 = 10
length_2 = 15
_, wcdm_LKL_mode_1, _ = synthetic_probability.solve_KL_ode(example.wcdm_shear_log_params_posterior_flow, example.wcdm_shear_log_params_prior_flow, y0, n=-1, length=length_1, num_points=1000)
_, wcdm_LKL_mode_2, _ = synthetic_probability.solve_KL_ode(example.wcdm_shear_log_params_posterior_flow, example.wcdm_shear_log_params_prior_flow, y0, n=-2, length=length_2, num_points=1000)

# local fisher:
wcdm_fisher = example.wcdm_shear_log_params_posterior_flow.metric(example.wcdm_shear_log_params_posterior_flow.cast([reference_point]))[0]
wcdm_prior_fisher = example.wcdm_shear_log_params_prior_flow.metric(example.wcdm_shear_log_params_prior_flow.cast([reference_point]))[0]
wcdm_eig, wcdm_eigv = utilities.KL_decomposition(wcdm_fisher, wcdm_prior_fisher)
# sort modes:
idx = np.argsort(wcdm_eig)[::-1]
eig = wcdm_eig[idx]
wcdm_eigv = wcdm_eigv[:, idx]


###############################################################################
# plot:

levels = [utilities.from_sigma_to_confidence(i) for i in range(2, 0, -1)]

x_size = 18.0
y_size = 9.0
main_fontsize = 10.0

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 3)
ax1 = plt.subplot(gs[0])
ax2 = plt.subplot(gs[1])
ax3 = plt.subplot(gs[2])

##############################################################################
# first panel LCDM:
##############################################################################

# obtain posterior on grid from samples:
density = example.posterior_chain_lcdm_shear.get2DDensity('omegam', 'sigma8', normalized=True)
_X1, _Y1 = np.meshgrid(density.x, density.y)
density.P = density.P / simps(simps(density.P, density.y), density.x)
levs = np.append(get_levels(density.P, density.x, density.y, levels), [np.amax(density.P)])

ax1.contour(_X1, _Y1, density.P, levs, linewidths=1., linestyles='-', colors=[colors[0] for i in levels])
cols = [colors[0] for i in levs]
cols[0] = tuple(list(cols[0])+[0.1])
cols[1] = tuple(list(cols[1])+[0.8])
ax1.contourf(_X1, _Y1, density.P, levs, colors=cols)

m1, m2 = np.exp(example.posterior_chain_lcdm_shear.getMeans([example.posterior_chain_lcdm_shear.index[name] for name in ['log_omegam', 'log_sigma8']]))
ax1.scatter(m1, m2, c=[colors[0]], edgecolors='white', zorder=999, s=20)

# plot modes:
temp = np.sqrt(lcdm_eig[-1])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
ax1.plot(m1*np.exp(alpha*lcdm_eigv[0, 0]), m2*np.exp(alpha*lcdm_eigv[1, 0]), c='k', lw=1., ls='-')
temp = np.sqrt(lcdm_eig[-2])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
ax1.plot(m1*np.exp(alpha*lcdm_eigv[0, 1]), m2*np.exp(alpha*lcdm_eigv[1, 1]), c='k', lw=1., ls='-')

# plot modes:
ax1.plot(np.exp(lcdm_LKL_mode_1[:, example.lcdm_shear_log_params_posterior_flow.param_names.index('log_omegam')]),
         np.exp(lcdm_LKL_mode_1[:, example.lcdm_shear_log_params_posterior_flow.param_names.index('log_sigma8')]),
         c=colors[1], lw=1., ls='-')
ax1.plot(np.exp(lcdm_LKL_mode_2[:, example.lcdm_shear_log_params_posterior_flow.param_names.index('log_omegam')]),
         np.exp(lcdm_LKL_mode_2[:, example.lcdm_shear_log_params_posterior_flow.param_names.index('log_sigma8')]),
         c=colors[2], lw=1., ls='-')


##############################################################################
# second panel mnu:
##############################################################################

# obtain posterior on grid from samples:
density = example.posterior_chain_mnu_shear.get2DDensity('omegam', 'sigma8', normalized=True)
_X1, _Y1 = np.meshgrid(density.x, density.y)
density.P = density.P / simps(simps(density.P, density.y), density.x)
levs = np.append(get_levels(density.P, density.x, density.y, levels), [np.amax(density.P)])

ax2.contour(_X1, _Y1, density.P, levs, linewidths=1., linestyles='-', colors=[colors[0] for i in levels])
cols = [colors[0] for i in levs]
cols[0] = tuple(list(cols[0])+[0.1])
cols[1] = tuple(list(cols[1])+[0.8])
ax2.contourf(_X1, _Y1, density.P, levs, colors=cols)

m1, m2 = np.exp(example.posterior_chain_mnu_shear.getMeans([example.posterior_chain_mnu_shear.index[name] for name in ['log_omegam', 'log_sigma8']]))
ax2.scatter(m1, m2, c=[colors[0]], edgecolors='white', zorder=999, s=20)

#ax2.plot(density.x, m2*(density.x/m1)**(0.55), ls='--', lw=1., color='k')
#ax2.plot(density.x, m2*(density.x/m1)**(-0.55), ls='--', lw=1., color='k')

# plot modes:
temp = np.sqrt(mnu_eig[-1])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
ax2.plot(m1*np.exp(alpha*mnu_eigv[0, 0]), m2*np.exp(alpha*mnu_eigv[1, 0]), c='k', lw=1., ls='-')
temp = np.sqrt(mnu_eig[-2])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
ax2.plot(m1*np.exp(alpha*mnu_eigv[0, 1]), m2*np.exp(alpha*mnu_eigv[1, 1]), c='k', lw=1., ls='-')

# plot modes:
ax2.plot(np.exp(mnu_LKL_mode_1[:, example.mnu_shear_log_params_posterior_flow.param_names.index('log_omegam')]),
         np.exp(mnu_LKL_mode_1[:, example.mnu_shear_log_params_posterior_flow.param_names.index('log_sigma8')]),
         c=colors[1], lw=1., ls='-')
ax2.plot(np.exp(mnu_LKL_mode_2[:, example.mnu_shear_log_params_posterior_flow.param_names.index('log_omegam')]),
         np.exp(mnu_LKL_mode_2[:, example.mnu_shear_log_params_posterior_flow.param_names.index('log_sigma8')]),
         c=colors[2], lw=1., ls='-')


##############################################################################
# third panel wcdm:
##############################################################################

# obtain posterior on grid from samples:
density = example.posterior_chain_wcdm_shear.get2DDensity('omegam', 'sigma8', normalized=True)
_X1, _Y1 = np.meshgrid(density.x, density.y)
density.P = density.P / simps(simps(density.P, density.y), density.x)
levs = np.append(get_levels(density.P, density.x, density.y, levels), [np.amax(density.P)])

ax3.contour(_X1, _Y1, density.P, levs, linewidths=1., linestyles='-', colors=[colors[0] for i in levels])
cols = [colors[0] for i in levs]
cols[0] = tuple(list(cols[0])+[0.1])
cols[1] = tuple(list(cols[1])+[0.8])
ax3.contourf(_X1, _Y1, density.P, levs, colors=cols)

m1, m2 = np.exp(example.posterior_chain_wcdm_shear.getMeans([example.posterior_chain_wcdm_shear.index[name] for name in ['log_omegam', 'log_sigma8']]))
ax3.scatter(m1, m2, c=[colors[0]], edgecolors='white', zorder=999, s=20)

# plot modes:
temp = np.sqrt(wcdm_eig[-1])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
ax3.plot(m1*np.exp(alpha*wcdm_eigv[0, 0]), m2*np.exp(alpha*wcdm_eigv[1, 0]), c='k', lw=1., ls='-')
temp = np.sqrt(wcdm_eig[-2])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
ax3.plot(m1*np.exp(alpha*wcdm_eigv[0, 1]), m2*np.exp(alpha*wcdm_eigv[1, 1]), c='k', lw=1., ls='-')

## plot analitic lines:
#ax3.plot(density.x, m2*(density.x/m1)**(0.55), ls='--', lw=1., color='k')
#ax3.plot(density.x, m2*(density.x/m1)**(-0.55), ls='--', lw=1., color='k')

# plot modes:
ax3.plot(np.exp(wcdm_LKL_mode_1[:, example.wcdm_shear_log_params_posterior_flow.param_names.index('log_omegam')]),
         np.exp(wcdm_LKL_mode_1[:, example.wcdm_shear_log_params_posterior_flow.param_names.index('log_sigma8')]),
         c=colors[1], lw=1., ls='-')
ax3.plot(np.exp(wcdm_LKL_mode_2[:, example.wcdm_shear_log_params_posterior_flow.param_names.index('log_omegam')]),
         np.exp(wcdm_LKL_mode_2[:, example.wcdm_shear_log_params_posterior_flow.param_names.index('log_sigma8')]),
         c=colors[2], lw=1., ls='-')


##############################################################################
# finalize plot:
##############################################################################

# title:
ax1.text(0.01, 1.03, 'a) $\\Lambda$CDM', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax1.transAxes)
ax2.text(0.01, 1.03, 'b) $\\Lambda$CDM + $m_{\\nu}$', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax2.transAxes)
ax3.text(0.01, 1.03, 'b) wCDM', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax3.transAxes)

# axes labels:
for _ax in [ax1, ax2, ax3]:
    _ax.set_xlabel('$\\Omega_m$', fontsize=main_fontsize)
ax1.set_ylabel('$\\sigma_8$', fontsize=main_fontsize)

# limits:
for _ax in [ax1, ax2, ax3]:
    _ax.set_xlim([0.1, 0.6])
    _ax.set_ylim([0.5, 1.2])

# ticks:
ticks = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
for _ax in [ax1, ax2, ax3]:
    _ax.set_xticks(ticks)
    _ax.set_xticklabels(ticks, fontsize=0.9*main_fontsize)
    _ax.get_xticklabels()[0].set_horizontalalignment('left')
    _ax.get_xticklabels()[-1].set_horizontalalignment('right')

ticks = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
ax1.set_yticks(ticks)
ax1.set_yticklabels(ticks, fontsize=0.9*main_fontsize)
ax1.get_yticklabels()[0].set_verticalalignment('bottom')
ax1.get_yticklabels()[-1].set_verticalalignment('top')
for _ax in [ax2, ax3]:
    _ax.set_yticks(ticks)
    _ax.set_yticklabels([], fontsize=0.9*main_fontsize)


plt.savefig(out_folder+'/figure_DES_Y3_shear.pdf')
plt.close('all')




pass
