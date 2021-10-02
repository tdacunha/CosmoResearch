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

@tf.function
def tf_KL_decomposition(matrix_a, matrix_b):
    """
    """
    # compute the eigenvalues of b, lambda_b:
    _lambda_b, _phi_b = tf.linalg.eigh(matrix_b)
    _sqrt_lambda_b = tf.linalg.diag(1./tf.math.sqrt(_lambda_b))
    _phib_prime = tf.matmul(_phi_b, _sqrt_lambda_b)
    #
    trailing_axes = [-1, -2]
    leading = tf.range(tf.rank(_phib_prime) - len(trailing_axes))
    trailing = trailing_axes + tf.rank(_phib_prime)
    new_order = tf.concat([leading, trailing], axis=0)
    _phib_prime_T = tf.transpose(_phib_prime, new_order)
    #
    _a_prime = tf.matmul(tf.matmul(_phib_prime_T, matrix_a), _phib_prime)
    _lambda, _phi_a = tf.linalg.eigh(_a_prime)
    _phi = tf.matmul(tf.matmul(_phi_b, _sqrt_lambda_b), _phi_a)
    return _lambda, _phi

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
y_size = 7.0
main_fontsize = 10.0

levels = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]
param_ranges = [[-2.2, -.6],[-.5,.4]]#np.log([[0.15, 0.5], [0.3, 1.5]])

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
result = example.posterior_flow.MAP_finder(disp=True)
maximum_posterior = result.x

#fisher_metric = example.posterior_flow.metric(example.posterior_flow.cast([maximum_posterior]))[0]
#prior_fisher_metric = example.prior_flow.metric(example.prior_flow.cast([maximum_posterior]))[0]

# get fisher from samples:
fisher_metric = np.linalg.inv(example.posterior_chain.cov())
prior_fisher_metric = np.linalg.inv(example.prior_chain.cov())
cov_metric = (example.posterior_chain.cov())
prior_cov_metric = (example.prior_chain.cov())
m1, m2 = maximum_posterior[0],maximum_posterior[1]#np.log10(.25),np.log10(.9)#example.posterior_chain.getMeans(pars = log_param_names)
# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])

ax1.contour(X, Y, P, analyze_2d_example.get_levels(P, x, y, levels), linewidths=1., zorder=-1., linestyles='-', colors=[color_utilities.nice_colors(6) for i in levels])

alpha = 100*np.linspace(-1, 1, 10000)
# plot KL:
eig, eigv = tf_KL_decomposition(prior_fisher_metric, fisher_metric)
#eig, eigv = tf_KL_decomposition(prior_cov_metric, cov_metric)
eig, eigv = eig.numpy(), eigv.numpy()
#eigv[:,1] = -eigv[:,1]

param_directions = eigv #np.linalg.inv(eigv.T) (when using covariance)
inds = (np.argsort(eig)[::-1])
param_directions_best = ((param_directions.T[inds]).T)#[:,:2]
mode = 0
norm0 = np.linalg.norm(eigv[:,0])
#ax1.plot(m1+alpha*param_directions_best[0,mode]/eig[mode], m2+alpha*param_directions_best[1,mode]/eig[mode], color='firebrick', lw=1.5, ls='-', marker = 'o',label='KL flow covariance')
#ax1.plot(maximum_posterior[0]+alpha*eigv[0,mode]/eig[mode], maximum_posterior[1]+alpha*eigv[1,mode]/eig[mode], lw=1.5, color='firebrick', ls='-', label='KL flow covariance')
ax1.axline(maximum_posterior, maximum_posterior+eigv[:, mode], lw=1, color=color_utilities.nice_colors(1), ls='-')

mode = 1
norm1 = np.linalg.norm(eigv[:,1])
#ax1.plot(m1+alpha*param_directions_best[0,mode]/eig[mode], m2+alpha*param_directions_best[1,mode]/eig[mode], lw=1.5, color='cadetblue', ls='-', marker = 'o')
ax1.axline(maximum_posterior, maximum_posterior+eigv[:, mode], lw=1, color=color_utilities.nice_colors(2), ls='-')

A = np.array([[1,-1],[0,1]])
fisher_metric_tilde = np.dot(np.dot((A.T), fisher_metric), (A))
prior_fisher_metric_tilde = np.dot(np.dot((A.T), prior_fisher_metric), (A))

eig, eigv = tf_KL_decomposition(prior_fisher_metric_tilde, fisher_metric_tilde)
eig, eigv = eig.numpy(), eigv.numpy()
eigv = np.dot(A,eigv)

alpha = np.linspace(-1, 1, 1000)
mode = 0
norm0 = np.linalg.norm(eigv[:,0])
#ax1.plot(maximum_posterior[0]+alpha*eigv[0, mode]/norm0, maximum_posterior[1]+alpha*eigv[1, mode]/norm0, lw=1.5, color='firebrick', ls=':', label='KL flow covariance')
ax1.axline(maximum_posterior, maximum_posterior+eigv[:, mode], lw=1.5, color=color_utilities.nice_colors(0), ls=':')
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
#ax1.plot(maximum_posterior[0]+alpha*eigv[0, mode]/norm1, maximum_posterior[1]+alpha*eigv[1, mode]/norm1, lw=1.5, color='cadetblue', ls=':')
ax1.axline(maximum_posterior, maximum_posterior+eigv[:, mode], lw=1.5, color=color_utilities.nice_colors(3), ls=':')

# limits:
#ax1.set_xlim([param_ranges[0][0], param_ranges[0][1]])
#ax1.set_ylim([-.6,.3])#(np.log([0.4, 1.4]))
ax1.set_xlim([-2.5, -0.5])
ax1.set_ylim([-0.6, 0.4])

# ticks:
ticks = [-2.5, -2.0, -1.5, -1.0, -0.5]
#[0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
ax1.set_xticks(ticks)
ax1.set_xticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_xticklabels()[0].set_horizontalalignment('left')
ax1.get_xticklabels()[-1].set_horizontalalignment('right')

ticks = [-0.6, -0.4, -0.2, 0.0, 0.2, 0.4]
#[0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
ax1.set_yticks(ticks)
ax1.set_yticklabels(ticks, fontsize=0.9*main_fontsize);
ax1.get_yticklabels()[0].set_verticalalignment('bottom')
ax1.get_yticklabels()[-1].set_verticalalignment('top')

# axes labels:
ax1.set_xlabel(r'$\theta_1$', fontsize=main_fontsize);
ax1.set_ylabel(r'$\theta_2$', fontsize=main_fontsize);

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
legend_labels = [r'$\mathcal{P}$', 'KLC of $\\theta$', 'KLC of $\\tilde{\\theta}$']

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
                bbox_to_anchor=(0.0, 0.02, 1.2, 0.9),
                )
leg.get_frame().set_linewidth('0.8')

# update dimensions:
bottom = .26#0.17
top = 0.99
left = 0.15
right = 0.99
wspace = 0.
hspace = 0.3
gs.update(bottom=bottom, top=top, left=left, right=right,
          wspace=wspace, hspace=hspace)
#leg.set_bbox_to_anchor( ( left, 0.005, right-left, right ) )
plt.savefig(out_folder+'/figure_2.pdf')
plt.close('all')
