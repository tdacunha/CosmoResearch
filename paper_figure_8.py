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
import tensorflow as tf

# add path for correct version of tensiometer:
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
from tensiometer import utilities
import synthetic_probability

###############################################################################
# initial settings:

import example_2_generate as example
import analyze_2d_example

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

###############################################################################
# plot:

# plot size in cm. Has to match to draft to make sure font sizes are consistent
x_size = 8.54
y_size = 7.0
main_fontsize = 10.0

levels = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]

#### compute LPCA flow in the first basis:

# compute maximum posterior and metric:
maximum_posterior = example.posterior_flow.MAP_coord

# compute the two base eigenvalues trajectories:
y0 = example.posterior_flow.cast(maximum_posterior)
length_1 = (example.posterior_flow.sigma_to_length(8)).astype(np.float32)
length_2 = (example.posterior_flow.sigma_to_length(8)).astype(np.float32)
ref_times_1, ref_start_1 = example.posterior_flow.solve_eigenvalue_ode_par(y0, n=0, length=length_1, num_points=200)
ref_times_2, ref_start_2 = example.posterior_flow.solve_eigenvalue_ode_par(y0, n=1, length=length_2, num_points=200)

#### get new distribution:

# change parameters:
new_samples = np.array([example.posterior_chain.samples[:, 0], example.posterior_chain.samples[:, 1] - example.posterior_chain.samples[:, 0]]).T
new_posterior_chain = MCSamples(samples=new_samples,
                            names=['theta_'+str(i+1) for i in range(len(example.param_names))],
                            labels=['\\theta_{'+str(i+1)+'}' for i in range(len(example.param_names))],
                            label='posterior')
# train:
new_posterior_flow = synthetic_probability.DiffFlowCallback(new_posterior_chain, param_names=new_posterior_chain.getParamNames().list(),
                                                            feedback=1, learning_rate=0.01, n_maf=example.n_maf, hidden_units=example.hidden_units)
# train:
new_posterior_flow.train(batch_size=example.batch_size, epochs=example.epochs, steps_per_epoch=example.steps_per_epoch, callbacks=example.callbacks)

# compute maximum posterior and metric:
new_maximum_posterior = np.array([maximum_posterior[0], maximum_posterior[1]-maximum_posterior[0]])

# compute the two base eigenvalues trajectories:
y0 = new_posterior_flow.cast(new_maximum_posterior)
length_1 = (new_posterior_flow.sigma_to_length(8)).astype(np.float32)
length_2 = (new_posterior_flow.sigma_to_length(8)).astype(np.float32)
new_ref_times_1, new_ref_start_1 = new_posterior_flow.solve_eigenvalue_ode_par(y0, n=0, length=length_1, num_points=200)
new_ref_times_2, new_ref_start_2 = new_posterior_flow.solve_eigenvalue_ode_par(y0, n=1, length=length_2, num_points=200)

# do the plot:
param_ranges = [[0.1, 0.5], [0.1, 1.1]]

# define the grid:
P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 400)
P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 400)
x, y = P1, P2
X, Y = np.meshgrid(x, y)

# compute probability:
log_P = new_posterior_flow.log_probability(np.array([X, Y], dtype=np.float32).T)
log_P = np.array(log_P).T
P = np.exp(log_P)
P = P / simps(simps(P, y), x)

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])

ax1.contour(X, Y, P, analyze_2d_example.get_levels(P, x, y, levels), linewidths=1., zorder=-1., linestyles='-', colors=[color_utilities.nice_colors(6) for i in levels])

# MAP:
ax1.scatter(*new_maximum_posterior, s=5.0, color='k', zorder=999)

# principal modes:
ax1.plot(*new_ref_start_1.numpy().T, lw=1.0, ls='-', color=color_utilities.nice_colors(0))
ax1.plot(*new_ref_start_2.numpy().T, lw=1.0, ls='-', color=color_utilities.nice_colors(1))

ax1.plot(ref_start_1[:, 0].numpy(), ref_start_1[:, 1].numpy()-ref_start_1[:, 0].numpy(), lw=1.5, ls=':', color=color_utilities.nice_colors(2))
ax1.plot(ref_start_2[:, 0].numpy(), ref_start_2[:, 1].numpy()-ref_start_2[:, 0].numpy(), lw=1.5, ls=':', color=color_utilities.nice_colors(3))

# limits:
ax1.set_xlim([0.1, 0.5])
ax1.set_ylim([0.1, 1.1])

# ticks:
ticks = [0.1, 0.2, 0.3, 0.4, 0.5]
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_xticklabels()[0].set_horizontalalignment('left')
ax1.get_xticklabels()[-1].set_horizontalalignment('right')

ticks = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1]
ax1.set_yticks(ticks)
ax1.set_yticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_yticklabels()[0].set_verticalalignment('bottom')
ax1.get_yticklabels()[-1].set_verticalalignment('top')

# axes labels:
ax1.set_xlabel(r'$\tilde{\theta}_1 = \theta_1$', fontsize=main_fontsize);
ax1.set_ylabel(r'$\tilde{\theta}_2 = \theta_2-\theta_1$', fontsize=main_fontsize);

# legend:
from matplotlib.legend_handler import HandlerBase
class object_1():
    pass
class AnyObjectHandler1(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], color=color_utilities.nice_colors(0), lw=1.)
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color=color_utilities.nice_colors(1), lw=1.)
        return [l1, l2]

class object_2():
    pass
class AnyObjectHandler2(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       x0, y0, width, height, fontsize, trans):
        l1 = plt.Line2D([x0,y0+width], [0.7*height,0.7*height], color=color_utilities.nice_colors(2), lw=1.5, ls=':')
        l2 = plt.Line2D([x0,y0+width], [0.3*height,0.3*height], color=color_utilities.nice_colors(3), lw=1.5, ls=':')
        return [l1, l2]

leg_handlers = [mlines.Line2D([], [], lw=1., ls='-', color='k'),
                object_1, object_2]
legend_labels = [r'$\mathcal{P}$', 'PCCs of $\\tilde{\\theta}$', 'PCCs of $\\theta$']

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
                loc = 'lower center',
                bbox_to_anchor=(0.0, 0.02, 1.2, 0.9),
                )
leg.get_frame().set_linewidth('0.8')

# update dimensions:
bottom = .26
top = 0.99
left = 0.15
right = 0.99
wspace = 0.
hspace = 0.3
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)
leg.set_bbox_to_anchor((left, 0.005, right-left, right))
plt.savefig(out_folder+'/figure_8.pdf')
plt.close('all')
