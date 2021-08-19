# import libraries:
import sys, os
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)

from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import getdist
getdist.chains.print_load_details = False
import scipy
import matplotlib.pyplot as plt
import IPython
from IPython.display import Markdown
import numpy as np
import seaborn as sns
# import the tensiometer tools that we need:
from tensiometer import utilities
from tensiometer import gaussian_tension
from tensiometer import mcmc_tension

import tensorflow as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
from scipy import optimize
from scipy.integrate import simps

import synthetic_probability
import importlib
importlib.reload(synthetic_probability)

import copy
import pickle

from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import tensiometer.gaussian_tension as gaussian_tension
from scipy import optimize

import example_2_generate as example

chain=example.posterior_chain
flow=example.posterior_flow
param_names=example.posterior_chain.getParamNames().list()
param_ranges=[[0.0, 0.6], [0.4, 1.5]]
outroot=example.out_folder+'posterior_'

prior_chain=example.prior_chain
prior_flow=example.prior_flow
prior_param_names=example.prior_chain.getParamNames().list()
prior_outroot=example.out_folder+'prior_'

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
        res = optimize.brentq(lambda x: _helper(x)-c, np.amin(P), np.amax(P))
        levs.append(res)
    return levs

# plotting preferences:
figsize = (8, 8)
fontsize = 15

# parameter ranges for plotting from the prior:
if param_ranges is None:
    param_ranges = np.array([np.amin(chain.samples, axis=0), np.amax(chain.samples, axis=0)]).T
    param_ranges = param_ranges[[chain.index[name] for name in param_names], :]

# parameter labels:
param_labels = [name.label for name in chain.getParamNames().parsWithNames(param_names)]
param_labels_latex = ['$'+name+'$' for name in param_labels]

# parameter grids:
P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 200)
P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 200)
x, y = P1, P2
X, Y = np.meshgrid(x, y)

# coarse parameter grid:
coarse_P1 = np.linspace(param_ranges[0][0], param_ranges[0][1], 20)
coarse_P2 = np.linspace(param_ranges[1][0], param_ranges[1][1], 20)
coarse_x, coarse_y = coarse_P1, coarse_P2
coarse_X, coarse_Y = np.meshgrid(coarse_x, coarse_y)

# obtain posterior on grid from samples:
density = chain.get2DDensity(param_names[0], param_names[1], normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
density.P = density.P / simps(simps(density.P, density.y), density.x)

# levels for contour plots:
levels_5 = [utilities.from_sigma_to_confidence(i) for i in range(5, 1, -1)]
levels_3 = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]

log_P = flow.log_probability(np.array([X, Y], dtype=np.float32).T)
log_P = np.array(log_P).T
P = np.exp(log_P)
P = P / simps(simps(P, y), x)



# feedback:

N = 10000
X_sample = np.array(flow.sample(N))
flow_chain = MCSamples(samples=X_sample,
                       #loglikes=-prior_flow.log_probability(X_sample).numpy(),
                       names=param_names,
                       label='Learned distribution')

g = plots.get_subplot_plotter()
g.triangle_plot([chain, flow_chain], params=param_names, filled=False)

X_sample = np.array(prior_flow.sample(N))
prior_flow_chain = MCSamples(samples=X_sample,
                       #loglikes=-prior_flow.log_probability(X_sample).numpy(),
                       names=param_names,
                       label='Learned distribution')

g = plots.get_subplot_plotter()
g.triangle_plot([prior_chain, prior_flow_chain], params=param_names, filled=False)
#g.export(outroot+'1_learned_posterior_distribution.pdf')
#plt.close('all')


# find the MAP in parameter space:
result = flow.MAP_finder(disp=True)
maximum_posterior = result.x
# mean:
mean = chain.getMeans([chain.index[name] for name in param_names])

cov_samples = chain.cov(pars=param_names)
prior_cov_samples = prior_chain.cov(pars=param_names)


# metrics from flow around mean:
covariance_metric = flow.metric(np.array([mean]).astype(np.float32))[0]
fisher_metric = flow.inverse_metric(np.array([mean]).astype(np.float32))[0]

prior_covariance_metric = prior_flow.metric(np.array([mean]).astype(np.float32))[0]
prior_fisher_metric = prior_flow.inverse_metric(np.array([mean]).astype(np.float32))[0]


###############################################################################

alpha = np.linspace(-1, 1, 1000)
plt.figure(figsize=figsize)

# plot KL of flow covariance metric
eig, eigv = utilities.KL_decomposition(prior_covariance_metric, covariance_metric)
print(eigv)
# inds = (np.argsort(eig)[::-1])
# param_directions = np.linalg.inv(eigv.T)
# eigv = (param_directions.T[inds]).T
print(eigv)
mode = 0
norm0 = np.linalg.norm(eigv[:,0])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm0, mean[1]+alpha*eigv[1, mode]/norm0, lw=1.5, color='k', ls='--', label='KL flow covariance', alpha = .5)
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm1, mean[1]+alpha*eigv[1, mode]/norm1, lw=1.5, color='k', ls='--',alpha = .5)

# plot KL of flow fisher metric
eig, eigv = utilities.KL_decomposition(prior_fisher_metric, fisher_metric)
inds = (np.argsort(eig)[::-1])
param_directions = np.linalg.inv(eigv.T)
eigv = (param_directions.T[inds]).T

mode = 0
norm0 = np.linalg.norm(eigv[:,0])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm0, mean[1]+alpha*eigv[1, mode]/norm0, lw=1., color='green', ls='-.', label='KL flow fisher',alpha = .5)
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm1, mean[1]+alpha*eigv[1, mode]/norm1, lw=1., color='green', ls='-.',alpha = .5)

# plot KL of covariance of samples
eig, eigv = utilities.KL_decomposition(prior_cov_samples, cov_samples)
inds = (np.argsort(eig)[::-1])
param_directions = np.linalg.inv(eigv.T)
eigv = (param_directions.T[inds]).T
mode = 0
norm0 = np.linalg.norm(eigv[:,0])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm0, mean[1]+alpha*eigv[1, mode]/norm0, lw=1., color='red', ls='--', label='KL samples',alpha = .5)
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
plt.plot(mean[0]+alpha*eigv[0, mode]/norm1, mean[1]+alpha*eigv[1, mode]/norm1, lw=1., color='red', ls='--',alpha = .5)

# plot PCA of flow covariance metric
_, eigv = np.linalg.eigh(covariance_metric)
mode = 0
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1.5, color='k', ls='--', label='PCA flow covariance')
mode = 1
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1.5, color='k', ls='--')

# plot PCA of flow fisher metric
_, eigv = np.linalg.eigh(fisher_metric)
mode = 0
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='green', ls='-.', label='PCA flow fisher')
mode = 1
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='green', ls='-.')

# # plot PCA of covariance of samples
# _, eigv = np.linalg.eigh(cov_samples)
# mode = 0
# plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='red', ls='--', label='samples')
# mode = 1
# plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], lw=1., color='red', ls='--')

# plot contours
plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
plt.scatter(mean[0], mean[1], color='k')

plt.legend()
plt.xlim([np.amin(P1), np.amax(P1)])
plt.ylim([np.amin(P2), np.amax(P2)])
plt.legend()
plt.xlabel(param_labels_latex[0], fontsize=fontsize)
plt.ylabel(param_labels_latex[1], fontsize=fontsize)
plt.tight_layout()
plt.show()

##########################################################################
# Plot of local metric eigenvectors
##########################################################################

r = np.linspace(-1, 1, 1000)

coords = np.array([coarse_X, coarse_Y], dtype=np.float32).reshape(2, -1).T
print(coords)
# compute the metric at all coordinates
local_metrics = flow.metric(coords)
prior_local_metrics = prior_flow.metric(coords)
print(local_metrics)
print(prior_local_metrics)

# compute the PCA eigenvalues and eigenvectors of each local metric
#PCA_eig, PCA_eigv = np.linalg.eigh(local_metrics)
#KL_eig, KL_eigv = utilities.KL_decomposition(prior_local_metrics, local_metrics)
KL_eig = np.zeros((400,2))
KL_eigv = np.zeros((400,2,2))
print(len(local_metrics))
for i in range(len(local_metrics)):

    KL_eig_i, KL_eigv_i = utilities.KL_decomposition(prior_local_metrics[i], local_metrics[i])
    KL_eig[i] = KL_eig_i
    norm  = np.linalg.norm(KL_eigv_i,axis = 1)
    norm_tile = np.tile(norm,(2,1)).T
    KL_eigv[i] = KL_eigv_i/norm_tile
    # if i == 0:
    #     print(KL_eigv_i)
    #     print(norm)
    #     print(KL_eigv_i/norm_tile)
    #     print(np.linalg.norm(KL_eigv_i/norm_tile,axis = 1))
#print(np.shape(KL_eigv))
# sort PCA so first mode is index 0
idx = np.argsort(KL_eig, axis=1)[0]
KL_eig = KL_eig[:, idx]
KL_eigv = KL_eigv[:, :, idx]

# plot PCA eigenvectors
mode = 0
plt.figure(figsize=figsize)
plt.quiver(coords[:, 0], coords[:, 1], KL_eigv[:, 0, mode], KL_eigv[:, 1, mode], color='red', angles='xy', label='First mode')
mode = 1
plt.quiver(coords[:, 0], coords[:, 1], KL_eigv[:, 0, mode], KL_eigv[:, 1, mode], color='cadetblue', angles='xy', label='Second mode')

# plot contours
plt.contour(X, Y, P, get_levels(P, x, y, levels_5), linewidths=1., linestyles='-', colors=['k' for i in levels_5])
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

# compute and plot eigenvectors of covariance of samples
eig, eigv = utilities.KL_decomposition(prior_cov_samples, cov_samples)
inds = (np.argsort(eig)[::-1])
param_directions = np.linalg.inv(eigv.T)
eigv = (param_directions.T[inds]).T
norm0 = np.linalg.norm(eigv[:,0])
mode = 0
plt.plot(maximum_posterior[0] + r*eigv[0, mode]/norm0, maximum_posterior[1] + r*eigv[1, mode]/norm0, ls='-', color='k')
mode = 1
norm1 = np.linalg.norm(eigv[:,1])
plt.plot(maximum_posterior[0] + r*eigv[0, mode]/norm1, maximum_posterior[1] + r*eigv[1, mode]/norm1, ls='-', color='k')

plt.legend()
plt.xlim([np.amin(P1), np.amax(P1)])
plt.ylim([np.amin(P2), np.amax(P2)])
plt.xlabel(param_labels_latex[0], fontsize=fontsize)
plt.ylabel(param_labels_latex[1], fontsize=fontsize)
plt.tight_layout()
plt.show()


def eigenvalue_ode(t, y, reference):
    """
    Solve the dynamical equation for eigenvalues.
    """
    # preprocess:
    x_par = tf.convert_to_tensor([tf.cast(y, tf.float32)])
    # map to original space to compute Jacobian (without inversion):
    #x_par = flow.map_to_original_coord(x)
    # precompute Jacobian and its derivative:
    # jac = flow.inverse_jacobian(x_par)[0]
    # jac_T = tf.transpose(jac)
    # jac_jac_T = tf.matmul(jac, jac_T)
    metric = flow.metric(x_par)
    prior_metric = prior_flow.metric(x_par)
    # compute eigenvalues:
    eig, eigv = utilities.KL_decomposition(prior_metric, metric)
    temp = tf.matmul(tf.transpose(eigv), tf.transpose([reference]))
    idx = tf.math.argmax(tf.abs(temp))[0]
    w = tf.convert_to_tensor([tf.math.sign(temp[idx]) * eigv[:, idx]])
    #
    return w

def solve_eigenvalue_ode_par(y0, n, length=1.5, num_points=100, **kwargs):
    """
    Solve eigenvalue problem in abstract space
    """
    # define solution points:
    solution_times = tf.linspace(0., length, num_points)
    # compute initial PCA:
    x_par = tf.convert_to_tensor([y0])
    #x_par = flow.map_to_original_coord(x_abs)
    # jac = flow.inverse_jacobian(x_par)[0]
    # jac_T = tf.transpose(jac)
    # jac_jac_T = tf.matmul(jac, jac_T)
    # compute eigenvalues:
    metric  = flow.metric(x_par)#[0]
    prior_metric = prior_flow.metric(x_par)#[0]
    eig, eigv = utilities.KL_decomposition(prior_metric, metric)

    # initialize solution:
    temp_sol_1 = np.zeros((num_points-1, flow.num_params))
    temp_sol_dot_1 = np.zeros((num_points-1, flow.num_params))
    temp_sol_2 = np.zeros((num_points-1, flow.num_params))
    temp_sol_dot_2 = np.zeros((num_points-1, flow.num_params))
    # integrate forward:
    solver = scipy.integrate.ode(eigenvalue_ode)
    solver.set_integrator('lsoda')
    solver.set_initial_value(y0, 0.)
    reference = eigv[:, n]
    for ind, t in enumerate(solution_times[1:]):
        # set the reference:
        solver.set_f_params(reference)
        # advance solver:
        yt = solver.integrate(t)
        # compute derivative after time-step:
        yprime = eigenvalue_ode(t, yt, reference)
        # update reference:
        reference = yprime[0]
        # save out:
        temp_sol_1[ind] = yt.copy()
        temp_sol_dot_1[ind] = yprime.numpy().copy()
    # integrate backward:
    solver = scipy.integrate.ode(eigenvalue_ode)
    #solver.set_integrator()
    solver.set_initial_value(y0, 0.)
    reference = - eigv[:, n]
    for ind, t in enumerate(solution_times[1:]):
        # set the reference:
        solver.set_f_params(reference)
        # advance solver:
        yt = solver.integrate(t)
        # compute derivative after time-step:
        yprime = eigenvalue_ode(t, yt, reference)
        # update reference:
        reference = yprime[0]
        # save out:
        temp_sol_2[ind] = yt.copy()
        temp_sol_dot_2[ind] = yprime.numpy().copy()
    # patch solutions:
    times = np.concatenate((-solution_times[::-1], solution_times[1:]))
    traj = np.concatenate((temp_sol_2[::-1], x_abs.numpy(), temp_sol_1))
    vel = np.concatenate((-temp_sol_dot_2[::-1], [eigv[:, n].numpy()], temp_sol_dot_1))
    #
    return times, traj, vel
#
# def solve_eigenvalue_ode_par(y0, n, length=1.5, num_points=100, **kwargs):
#     """
#     Solve eigenvalue ODE in parameter space
#     """
#     # go to abstract space:
#     x_par = tf.convert_to_tensor([y0])
#     x_abs = flow.map_to_abstract_coord(x_par)[0]
#     # call solver:
#     times, traj, vel = solve_eigenvalue_ode_abs(x_abs, n, length=length, num_points=num_points, **kwargs)
#     # convert back:
#     traj = flow.map_to_original_coord(tf.cast(traj, tf.float32))
#     #
#     return times, traj

# lines along the global principal components:
y0 = maximum_posterior.astype(np.float32)
length = (flow.sigma_to_length(6)).astype(np.float32)

_, start_1 = solve_eigenvalue_ode_par([y0], n=0, length=length, num_points=5)
_, start_0 = solve_eigenvalue_ode_par([y0], n=1, length=length, num_points=5)

# solve:
modes_0, modes_1 = [], []
for start in start_0:
    _, mode = solve_eigenvalue_ode_par(start, n=0, length=length, num_points=100)
    modes_0.append(mode)
for start in start_1:
    _, mode = solve_eigenvalue_ode_par(start, n=1, length=length, num_points=100)
    modes_1.append(mode)

# plot:
plt.figure(figsize=(2*figsize[0], figsize[1]))
gs = gridspec.GridSpec(1, 2)
ax1 = plt.subplot(gs[0, 0])
ax2 = plt.subplot(gs[0, 1])

for mode in modes_0:
    ax1.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='k')
for mode in modes_1:
    ax1.plot(mode[:, 0], mode[:, 1], lw=1., ls='-', color='red')

ax1.contour(X, Y, P, get_levels(P, x, y, levels_3), linewidths=2., linestyles='-', colors=['blue' for i in levels_5], zorder=999)
ax1.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

ax1.set_xlim([np.amin(P1), np.amax(P1)])
ax1.set_ylim([np.amin(P2), np.amax(P2)])
ax1.set_xlabel(param_labels_latex[0], fontsize=fontsize)
ax1.set_ylabel(param_labels_latex[1], fontsize=fontsize)

for mode in modes_0:
    mode_abs = flow.map_to_abstract_coord(mode)
    ax2.plot(*np.array(mode_abs).T, lw=1., ls='-', color='k')
for mode in modes_1:
    mode_abs = flow.map_to_abstract_coord(mode)
    ax2.plot(*np.array(mode_abs).T, lw=1., ls='-', color='red')

plt.tight_layout()
plt.show()
