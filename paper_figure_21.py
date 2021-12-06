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
import synthetic_probability
from tensorflow_probability import bijectors as bj

# import example:
#import example_CMB_lensing as example
import synthetic_probability

###############################################################################
# initial settings:

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# color palette:
colors = [color_utilities.nice_colors(i) for i in range(6)]
line_colors = [1,0,2]
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
# get chains:

prior_chain_lcdm_CMB_lensing = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'008_CMB_lensing_prior', no_cache=True, settings=DES_generate.settings)
posterior_chain_lcdm_CMB_lensing = getdist.mcsamples.loadMCSamples(file_root=DES_generate.chains_dir+'008_CMB_lensing', no_cache=True, settings=DES_generate.settings)

# add log parameters:
for ch in [prior_chain_lcdm_CMB_lensing, posterior_chain_lcdm_CMB_lensing]:
    temp_names = ch.getParamNames().list()
    for name in temp_names:
        if np.all(ch.samples[:, ch.index[name]] > 0.):
            ch.addDerived(np.log(ch.samples[:, ch.index[name]]), name='log_'+name, label='\\log '+ch.getParamNames().parWithName(name).label)
    # update after adding all parameters:
    ch.updateBaseStatistics()

###############################################################################
# get flows:

param_names = ['omegam', 'sigma8', 'omegab', 'H0', 'ns']

#params_flow_cache = './results/example_CMB_lensing_nobound/'+'lcdm_CMB_lensing_params_flow_cache_selected'
params_flow_cache = './results/example_CMB_lensing_nobound/'+'lcdm_CMB_lensing_params_flow_cache'
temp = DES_generate.helper_load_chains(param_names, prior_chain_lcdm_CMB_lensing, posterior_chain_lcdm_CMB_lensing, params_flow_cache, prior_bijector=None)
lcdm_CMB_lensing_params_prior_flow, lcdm_CMB_lensing_params_posterior_flow = temp

transformation = [bj.Log()]*len(param_names)
lcdm_CMB_lensing_log_params_prior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_params_prior_flow, transformation)
lcdm_CMB_lensing_log_params_posterior_flow = synthetic_probability.TransformedDiffFlowCallback(lcdm_CMB_lensing_params_posterior_flow, transformation)

posterior_chain = posterior_chain_lcdm_CMB_lensing
prior_chain = prior_chain_lcdm_CMB_lensing
posterior_flow = lcdm_CMB_lensing_log_params_posterior_flow
prior_flow = lcdm_CMB_lensing_log_params_prior_flow

###############################################################################
# add derived parameters to chain:

for chain in [prior_chain, posterior_chain]:
    p = chain.getParams()
    param = p.omegam**0.8 * (p.H0 / 100.)
    chain.addDerived(param, 'p_1', 'h\\Omega_m^{0.8}')
    chain.updateBaseStatistics()

###############################################################################
# get linear and non-linear PCA:

# get param names:
param_names = posterior_flow.param_names
num_params = len(param_names)

# get reference point:
reference_point = posterior_chain.getMeans(pars=[posterior_chain.index[name] for name in param_names])

# get local fisher:
fisher = posterior_flow.metric(posterior_flow.cast([reference_point]))[0]
prior_fisher = prior_flow.metric(prior_flow.cast([reference_point]))[0]

# do local KL decomposition:
eig, eigv = utilities.KL_decomposition(fisher, prior_fisher)
sqrt_fisher = scipy.linalg.sqrtm(fisher)
# sort modes:
idx = np.argsort(eig)[::-1]
eig = eig[idx]
eigv = eigv[:, idx]

# solve for non-linear modes:
y0 = posterior_flow.cast(reference_point)
length_1 = 40
length_2 = 10
options = {'name':'vode', 'method':'bdf', 'order':1}
_, LKL_mode_1, _ = synthetic_probability.solve_KL_ode(posterior_flow, prior_flow, y0, n=-1, length=length_1, num_points=50, integrator_options=options)
_, LKL_mode_2, _ = synthetic_probability.solve_KL_ode(posterior_flow, prior_flow, y0, n=-2, length=length_2, num_points=50, integrator_options=options)

###############################################################################
# make the figure:

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# plot size in cm. Has to match to draft to make sure font sizes are consistent
x_size = 8.54
y_size = 7.5
main_fontsize = 10.0

levels = [utilities.from_sigma_to_confidence(i) for i in range(2, 0, -1)]

# obtain posterior on grid from samples:
density_1 = posterior_chain.get2DDensity('p_1', 'sigma8', normalized=True, smooth_scale_2D=0.5)
_X1, _Y1 = np.meshgrid(density_1.x, density_1.y)
density_1.P = density_1.P / simps(simps(density_1.P, density_1.y), density_1.x)

density_2 = prior_chain.get2DDensity('p_1', 'sigma8', normalized=True, smooth_scale_2D=0.5)
_X2, _Y2 = np.meshgrid(density_2.x, density_2.y)
density_2.P = density_2.P / simps(simps(density_2.P, density_2.y), density_2.x)

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])

# pdfs:
ax1.contour(_X1, _Y1, density_1.P, get_levels(density_1.P, density_1.x, density_1.y, levels), linewidths=1., linestyles='-', colors=[colors[3] for i in levels])
ax1.contour(_X2, _Y2, density_2.P, get_levels(density_2.P, density_2.x, density_2.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels])

levs = np.append(get_levels(density_2.P, density_2.x, density_2.y, levels), [np.amax(density_2.P)])
cols = [(0, 0, 0) for i in levs]
cols[0] = tuple(list(cols[0])+[0.05])
cols[1] = tuple(list(cols[1])+[0.1])
ax1.contourf(_X2, _Y2, density_2.P, levs, colors=cols)

levs = np.append(get_levels(density_1.P, density_1.x, density_1.y, levels), [np.amax(density_1.P)])
cols = [colors[3] for i in levs]
cols[0] = tuple(list(cols[0])+[0.1])
cols[1] = tuple(list(cols[1])+[0.8])
ax1.contourf(_X1, _Y1, density_1.P, levs, colors=cols)

# reference point:
m1 = np.exp(reference_point[param_names.index('log_omegam')])**0.8*(np.exp(reference_point[param_names.index('log_H0')]) / 100.)
m2 = np.exp(reference_point[param_names.index('log_sigma8')])
ax1.scatter(m1, m2, c=[colors[3]], edgecolors='white', zorder=999, s=20)

# non-linear modes:
mode_1_x = np.exp(LKL_mode_1[:, param_names.index('log_omegam')])**0.8*(np.exp(LKL_mode_1[:, param_names.index('log_H0')]) / 100.)
mode_1_y = np.exp(LKL_mode_1[:, param_names.index('log_sigma8')])
ax1.plot(mode_1_x, mode_1_y, c=colors[1], lw=1., ls='-')
mode_2_x = np.exp(LKL_mode_2[:, param_names.index('log_omegam')])**0.8*(np.exp(LKL_mode_2[:, param_names.index('log_H0')]) / 100.)
mode_2_y = np.exp(LKL_mode_2[:, param_names.index('log_sigma8')])
ax1.plot(mode_2_x, mode_2_y, c=colors[0], lw=1., ls='-')

# linear modes:
mode = 0
temp = np.sqrt(eig[mode])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
_omegam = np.exp(reference_point[param_names.index('log_omegam')] + alpha*eigv[param_names.index('log_omegam'), mode])
_sigma8 = np.exp(reference_point[param_names.index('log_sigma8')] + alpha*eigv[param_names.index('log_sigma8'), mode])
_h = np.exp(reference_point[param_names.index('log_H0')] + alpha*eigv[param_names.index('log_H0'), mode]) / 100.
ax1.plot(_omegam**0.8*_h, _sigma8, c=colors[1], lw=1., ls='--')

mode = 1
temp = np.sqrt(eig[mode])
alpha = 200.*np.linspace(-1./temp, 1./temp, 1000)
_omegam = np.exp(reference_point[param_names.index('log_omegam')] + alpha*eigv[param_names.index('log_omegam'), mode])
_sigma8 = np.exp(reference_point[param_names.index('log_sigma8')] + alpha*eigv[param_names.index('log_sigma8'), mode])
_h = np.exp(reference_point[param_names.index('log_H0')] + alpha*eigv[param_names.index('log_H0'), mode]) / 100.
ax1.plot(_omegam**0.8*_h, _sigma8, c=colors[0], lw=1., ls='--')

# limits:
ax1.set_xlim([0.1, 0.4])
ax1.set_ylim([0.4, 1.0])

# ticks:
ticks = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_xticklabels()[0].set_horizontalalignment('left')
ax1.get_xticklabels()[-1].set_horizontalalignment('right')

ticks = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ax1.set_yticks(ticks)
ax1.set_yticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_yticklabels()[0].set_verticalalignment('bottom')
ax1.get_yticklabels()[-1].set_verticalalignment('top')

# axes labels:
ax1.set_xlabel(r'$\Omega_m^{0.8} h$', fontsize=main_fontsize);
ax1.set_ylabel(r'$\sigma_8$', fontsize=main_fontsize);

# title:
ax1.text(0.01, 1.03, 'CMB lensing', verticalalignment='bottom', horizontalalignment='left', fontsize=main_fontsize, transform=ax1.transAxes)

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

leg_handlers = [mpatches.Patch(color=list((0, 0, 0))+[0.1], ec='k'),
                mpatches.Patch(color=list(colors[3])+[0.8], ec=colors[3]),
                object_1, object_2]

legend_labels = [
                 'Prior',
                 'CMB lensing',
                 'Lin. CPCC',
                 'Non-lin. CPCC',
                 ]

# legend for the second plot:
leg = fig.legend(handles=leg_handlers,
                 labels=legend_labels,
                 handler_map={object_1: AnyObjectHandler1(), object_2: AnyObjectHandler2()},
                 fontsize=0.9*main_fontsize,
                 frameon=True,
                 fancybox=False,
                 edgecolor='k',
                 ncol=2,
                 borderaxespad=0.0,
                 columnspacing=0.7,
                 handlelength=1.5,
                 handletextpad=0.3,
                 loc = 'lower center',
                 bbox_to_anchor=(0.0, 0.02, 1.2, 0.9),
                 )
leg.get_frame().set_linewidth('0.8')
leg.get_title().set_fontsize(main_fontsize)

# update dimensions:
bottom = 0.30
top = 0.92
left = 0.14
right = .99
wspace = 0.
hspace = 0.3
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)
leg.set_bbox_to_anchor((left, 0.002, right-left, right))

plt.savefig(out_folder+'/figure_21.pdf')
plt.close('all')
