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
import flow_copy
from scipy import optimize
from scipy.integrate import simps

# load the chains (remove no burn in since the example chains have already been cleaned):
chains_dir = here+'/tensiometer/test_chains/'
# the data chain:
settings = {'ignore_rows':0, 'smooth_scale_1D':0.3, 'smooth_scale_2D':0.3}
chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'DES', no_cache=True, settings=settings)
#chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'Planck18TTTEEE', no_cache=True, settings=settings)
# the prior chain:
prior_chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'prior', no_cache=True, settings=settings)

# select parameters:
param_names = ['omegam', 'sigma8']

# define the flow:
#flow_callback = mcmc_tension.DiffFlowCallback(chain, param_names=param_names, feedback=1, learning_rate=0.01)
flow_callback = flow_copy.DiffFlowCallback(chain, param_names=param_names, feedback=1, learning_rate=0.01)


# train:
from tensorflow.keras.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau()]

batch_size = 8192
epochs = 40
steps_per_epoch = 128

flow_callback.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

# plot learned distribution from samples:
N = 10000
X_sample = np.array(flow_callback.dist_learned.sample(N))
Y_sample = np.array(flow_callback.dist_gaussian_approx.sample(N))
flow_chain = MCSamples(samples=X_sample, names=param_names, label='Learned distribution')
Y_chain = MCSamples(samples=Y_sample, names=param_names, label='Transformed distribution')

g = plots.get_subplot_plotter()
g.triangle_plot([chain, flow_chain, Y_chain], params=param_names, filled=False)
###############################################################################
# plot learned distribution from value in different ways:
###############################################################################


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
    #
    return levs


omegam = np.linspace(.1, .5, 100)
sigma8 = np.linspace(.6, 1.2, 100)

x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)

# using the method implemented in dist_learned:
log_P = flow_callback.dist_learned.log_prob(np.array([X, Y], dtype=np.float32).T)
log_P = np.array(log_P).T
P = np.exp(log_P)
P = P / simps(simps(P, y), x)

# doing the calculation directly:
abs_X = flow_callback.Z2X_bijector.inverse(np.array([X, Y], dtype=np.float32).T)
gaussian = tfd.MultivariateNormalDiag(np.zeros(2, dtype=np.float32), np.ones(2, dtype=np.float32))
log_P_2 = gaussian.log_prob(abs_X) - flow_callback.Z2X_bijector.forward_log_det_jacobian(abs_X, event_ndims=1)
log_P_2 = np.array(log_P_2).T
P2 = np.exp(log_P_2)
P2 = P2 / simps(simps(P2, y), x)

# plot:
levels = [0.95, 0.68]
plt.contour(X, Y, P, get_levels(P, x, y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels])
plt.contour(X, Y, P2, get_levels(P2, x, y, levels), linewidths=1., linestyles='-', colors=['green' for i in levels])
density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='--', colors=['red' for i in levels])

###############################################################################
# find maximum posterior:
###############################################################################

from scipy.optimize import differential_evolution

# MAP:
bounds = [[.1, .5], [.6, 1.2]]
result = differential_evolution(lambda x: -flow_callback.dist_learned.log_prob(np.array(x, dtype=np.float32)), bounds, disp=True)
print(result)
maximum_posterior = result.x
# find where the MAP goes:
map_image = flow_callback.Z2X_bijector.inverse(np.array(maximum_posterior, dtype=np.float32))
print(maximum_posterior, np.array(map_image))

# mean:
mean = chain.getMeans([chain.index[name] for name in ['omegam', 'sigma8']])
mean_image = flow_callback.Z2X_bijector.inverse(np.array(mean, dtype=np.float32))
print(mean, np.array(mean_image))

# plot:
density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='--', colors=['red' for i in levels])
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='red', label='MAP')
plt.scatter(mean[0], mean[1], color='green', label='mean')
plt.legend()

###############################################################################
# get covariance from samples and from flow:
###############################################################################

# covariance from samples
cov_samples = chain.cov(pars=['omegam', 'sigma8'])
print('Covariance', cov_samples)
fisher_samples = np.linalg.inv(cov_samples)
print('Fisher', fisher_samples)

# covariance from flow around mean:
x = tf.constant(np.array(mean_image)) # or tf.constant, Variable but Variable doesn;t work with tf.function
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g: #persistent true doesn't seem to affect rn
    g.watch(x)
    y = flow_callback.Z2X_bijector(x)
jac = g.jacobian(y, x)
mu = np.identity(2)
covariance_metric = np.dot(np.array(jac), np.dot(mu, np.array(jac).T))

x = tf.constant(np.array(mean).astype(np.float32))
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g: #persistent true doesn't seem to affect rn
    g.watch(x)
    y = flow_callback.Z2X_bijector.inverse(x)
jac = g.jacobian(y, x)
mu = np.identity(2)
fisher_metric = np.dot(np.dot(np.array(jac).T, mu), np.array(jac))

# compare:
alpha = np.linspace(-1, 1, 1000)
_, eigv = np.linalg.eigh(covariance_metric)
mode = 0
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='k', ls='--', label='flow covariance')
mode = 1
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='k', ls='--')

alpha = np.linspace(-1, 1, 1000)
_, eigv = np.linalg.eigh(fisher_metric)
mode = 0
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='green', ls='-.', label='flow fisher')
mode = 1
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='green', ls='-.')

alpha = np.linspace(-1, 1, 1000)
_, eigv = np.linalg.eigh(cov_samples)
mode = 0
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='red', ls='-', label='samples')
mode = 1
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='red', ls='-')


density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='--', colors=['red' for i in levels])
plt.scatter(mean[0], mean[1], color='k')

plt.xlim([0.15, 0.4])
plt.ylim([0.6, 1.2])
plt.legend()

# covariance from flow around map:
x = tf.constant(np.array(map_image))
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g: #persistent true doesn't seem to affect rn
    g.watch(x)
    y = flow_callback.Z2X_bijector(x)
jac = g.jacobian(y, x)
mu = np.identity(2)
covariance_metric = np.dot(np.array(jac), np.dot(mu, np.array(jac).T))

x = tf.constant(np.array(maximum_posterior).astype(np.float32))
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g: #persistent true doesn't seem to affect rn
    g.watch(x)
    y = flow_callback.Z2X_bijector.inverse(x)
jac = g.jacobian(y, x)
mu = np.identity(2)
fisher_metric = np.dot(np.dot(np.array(jac).T, mu), np.array(jac))

# compare:
alpha = np.linspace(-1, 1, 1000)
_, eigv = np.linalg.eigh(covariance_metric)
mode = 0
plt.plot(maximum_posterior[0]+alpha*eigv[0, mode], maximum_posterior[1]+alpha*eigv[1, mode], color='k', ls='--', label='flow covariance')
mode = 1
plt.plot(maximum_posterior[0]+alpha*eigv[0, mode], maximum_posterior[1]+alpha*eigv[1, mode], color='k', ls='--')

alpha = np.linspace(-1, 1, 1000)
_, eigv = np.linalg.eigh(fisher_metric)
mode = 0
plt.plot(maximum_posterior[0]+alpha*eigv[0, mode], maximum_posterior[1]+alpha*eigv[1, mode], color='green', ls='-.', label='flow fisher')
mode = 1
plt.plot(maximum_posterior[0]+alpha*eigv[0, mode], maximum_posterior[1]+alpha*eigv[1, mode], color='green', ls='-.')

alpha = np.linspace(-1, 1, 1000)
_, eigv = np.linalg.eigh(cov_samples)
mode = 0
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='red', ls='-', label='samples')
mode = 1
plt.plot(mean[0]+alpha*eigv[0, mode], mean[1]+alpha*eigv[1, mode], color='red', ls='-')


density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='--', colors=['red' for i in levels])
plt.scatter(mean[0], mean[1], color='k')
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='red')

plt.xlim([0.15, 0.4])
plt.ylim([0.6, 1.2])
plt.legend()

###############################################################################
# trace geodesics passing from the maximum posterior
###############################################################################

import matplotlib


r = np.linspace(0.0, 20.0, 1000)
theta = np.linspace(0.0, 2.0*np.pi, 30)

# compute geodesics:
geodesics = []
for t in theta:
    geo = np.array([map_image[0] + r*np.cos(t),
                    map_image[1] + r*np.sin(t)], dtype=np.float32)
    geodesics.append(flow_callback.Z2X_bijector(geo.T))

# geodesics aligned with abstract coordinate axes:
r = np.linspace(-20.0, 20.0, 1000)

t = 0.0
geo = np.array([map_image[0] + r*np.cos(t),
                map_image[1] + r*np.sin(t)], dtype=np.float32)
geo_1 = flow_callback.Z2X_bijector(geo.T)

t = np.pi/2.
geo = np.array([map_image[0] + r*np.cos(t),
                map_image[1] + r*np.sin(t)], dtype=np.float32)
geo_2 = flow_callback.Z2X_bijector(geo.T)

# plot:
cmap = matplotlib.cm.get_cmap('Spectral')
for ind, geo in enumerate(geodesics):
    plt.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)

plt.plot(*np.array(geo_1).T, color='k', ls='--', zorder=-10, label='$\\theta=0$')
plt.plot(*np.array(geo_2).T, color='k', ls='-.', zorder=-10, label='$\\theta=\\pi/2$')

density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels], zorder=0)
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')
plt.xlim([0.15, 0.4])
plt.ylim([0.6, 1.2])
plt.legend()


###############################################################################
# asyntotic structure:
###############################################################################

r = np.linspace(0.0, 1000.0, 1000)
theta = np.linspace(0.0, 2.0*np.pi, 100)

# copmpute PCA:
eig, eigv = np.linalg.eigh(covariance)

# compute geodesics:
geodesics = []
for t in theta:
    geo = np.array([map_image[0] + r*np.cos(t),
                    map_image[1] + r*np.sin(t)], dtype=np.float32)
    geodesics.append(flow_callback.Z2X_bijector(geo.T))

# geodesics aligned with abstract coordinate axes:
r = np.linspace(-1000.0, 1000.0, 1000)

t = 0.0
geo = np.array([map_image[0] + r*np.cos(t),
                map_image[1] + r*np.sin(t)], dtype=np.float32)
geo_1 = flow_callback.Z2X_bijector(geo.T)

t = np.pi/2.
geo = np.array([map_image[0] + r*np.cos(t),
                map_image[1] + r*np.sin(t)], dtype=np.float32)
geo_2 = flow_callback.Z2X_bijector(geo.T)

# plot:
cmap = matplotlib.cm.get_cmap('Spectral')
for ind, geo in enumerate(geodesics):
    plt.plot(*np.array(geo).T, color=cmap(ind/len(geodesics)), zorder=-10)

plt.plot(*np.array(geo_1).T, color='k', ls='--', zorder=-10, label='$\\theta=0$')
plt.plot(*np.array(geo_2).T, color='k', ls='-.', zorder=-10, label='$\\theta=\\pi/2$')

mode = 0
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')
mode = 1
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')

density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels], zorder=0)
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

plt.xlim([-10, 10.0])
plt.ylim([-10, 10.0])
plt.legend()


###############################################################################
# MR arrived here
###############################################################################



# Experimenting with metric field plot:
#from tensorflow_probability.python.math import gradient
#tf.function(experimental_relax_shapes=True)
#@tf.function(experimental_relax_shapes=True)
#X2Z_bijector = tfb.Invert(flow_callback.Z2X_bijector)
#Y2Z_bijector = tfb.Invert(flow_callback.Z2Y_bijector)

p = plots.get_subplot_plotter(subplot_size = 5)
p.plot_2d([chain, flow_chain], param1=param_names[0], param2 = param_names[1], filled=False)
ax = p.subplots[0,0]

omegam = np.linspace(.1, .5, 10)
sigma8 = np.linspace(.6, 1.2, 10)

x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)

#abs_X = flow_callback.Z2X_bijector.inverse(np.array([X, Y], dtype=np.float32).T)
#abs_X = np.array(abs_X)

for om in omegam:
    for sig in sigma8:
        P1 = np.array([om, sig])

        P1_prime = np.array(flow_callback.Z2X_bijector.inverse(P1.astype(np.float32)))

        #P1_primeprime = np.array(flow_callback.Z2Y_bijector(P1_prime.astype(np.float32)))
        #print(P1,P1_primeprime)
        #_,grads = gradient.value_and_gradient(flow_callback.Z2X_bijector, P1_prime)
        #_1,jac = gradient.value_and_batch_jacobian(flow_callback.Z2X_bijector(x), x)

        x = tf.constant(P1_prime.astype(np.float32)) # or tf.constant, Variable but Variable doesn;t work with tf.function
        #delta = tf.Variable([0.0,0.0])
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g: #persistent true doesn't seem to affect rn
        #g = tf.GradientTape(persistent=True)
            g.watch(x)
            y = flow_callback.Z2X_bijector(x)
        jac = g.jacobian(y, x)

        #print(np.array(jac))
        #print(type(jac))
        mu = np.identity(2)
        #metric = np.matmul((np.array(jac)),np.matmul(mu,(np.array(jac)).T))
        metric = np.dot((np.array(jac)),np.dot(mu,(np.array(jac).T)))
        #print(metric)
        PCA_eig, PCA_eigv = np.linalg.eigh(metric)
        # sort:
        idx = np.argsort(PCA_eig)[::-1]
        PCA_eig = PCA_eig[idx]
        PCA_eigv = PCA_eigv[:, idx]
        #print(PCA_eigv)
        #print(PCA_eig) #second value is first mode?
        mode = 0
        ax.quiver(P1[0],P1[1],PCA_eigv[0,mode],PCA_eigv[1,mode], color = 'cadetblue')

plt.savefig('test.png')
plt.show()






]
# Inverted plot in abstract space:

X2Z_bijector = tfb.Invert(flow_callback.Z2X_bijector)
num_params = flow_callback.num_params
dist_learned_inverted = tfd.TransformedDistribution(distribution=flow_callback.dist_learned, bijector=X2Z_bijector)

#Z_sample = np.array(flow_callback.dist_gaussian_approx.sample(N)) #Using Y2Xbijector
#Z_sample = np.array(flow_callback.dist_transformed.sample(N)) #Using Z2Ybijector
Z_sample = np.array(dist_learned_inverted.sample(N))

flow_chain1 = MCSamples(samples=Z_sample, names=['Z1', 'Z2'], label='Transformed distribution')

g = plots.get_subplot_plotter()
g.triangle_plot([flow_chain1], params=['Z1', 'Z2'], filled=False)

# slice two dimensions:
#x = np.linspace(0.0, 2.0, 100)
dir_1 = (np.vstack((x, np.zeros(len(x)))).T.astype(np.float32))
dir_2 = (np.vstack((np.zeros(len(x)), x)).T.astype(np.float32))
ax = g.subplots[1, 0]
ax.plot(dir_1[:, 0], dir_1[:, 1])
ax.plot(dir_2[:, 0], dir_2[:, 1])

# Making gif
# Inverted plot in abstract space:
X2Z_bijector = tfb.Invert(flow_callback.Z2X_bijector)
num_params = flow_callback.num_params
dist_learned_inverted = tfd.TransformedDistribution(distribution=flow_callback.dist_learned, bijector=X2Z_bijector)

#X_sample1 = np.array(flow_callback.dist_transformed.sample(N))
Z_sample = np.array(dist_learned_inverted.sample(N))

flow_chain1 = MCSamples(samples=Z_sample, names=['Z1', 'Z2'], label='Transformed distribution')



# Plot in original space:



x = np.linspace(0.0, 100.0, 500) #100
#print(np.linspace(0,2*np.pi,8, endpoint = True))
for n,theta in enumerate(np.linspace(0,2*np.pi,24, endpoint = True)):

    vec_1 = np.vstack((x*np.cos(theta), -x*np.sin(theta))).T
    vec_2 = np.vstack((x*np.sin(theta), x*np.cos(theta))).T

    dir_1 = vec_1.astype(np.float32)
    dir_2 = vec_2.astype(np.float32)
    g1 = plots.get_subplot_plotter(subplot_size = 5)
    g1.plot_2d([flow_chain1], param1='Z1', param2 = 'Z2', filled=False)

    ax1 = g1.subplots[0,0]
    ax1.plot(dir_1[:, 0], dir_1[:, 1])
    ax1.plot(dir_2[:, 0], dir_2[:, 1])
    #g1.export('/Users/TaraD/Desktop/ParameterGif/ZPlot_%s' % (n))


    dir_1 = flow_callback.Z2X_bijector(vec_1.astype(np.float32))
    dir_2 = flow_callback.Z2X_bijector(vec_2.astype(np.float32))
    #dir_3 = flow_callback.Z2X_bijector(np.vstack((-x, np.zeros(len(x)))).T.astype(np.float32))
    g = plots.get_subplot_plotter(subplot_size = 5)
    g.plot_2d([chain, flow_chain], param1=param_names[0], param2 = param_names[1], filled=False)
    ax = g.subplots[0,0]
    ax.plot(dir_1[:, 0], dir_1[:, 1])
    ax.plot(dir_2[:, 0], dir_2[:, 1])

    g1.export('/Users/TaraD/Desktop/ParameterGif/ZPlot_%s.png' % (n))

    g.export('/Users/TaraD/Desktop/ParameterGif/XPlot_%s.png' % (n))





#g = plots.get_subplot_plotter()
#g.triangle_plot([chain, flow_chain], params=param_names, filled=False)
# plt.figure()
# plt.plot(dir_1[:, 0], dir_1[:, 1])
# plt.plot(dir_2[:, 0], dir_2[:, 1])

# dir_1 = flow_callback.Y2X_bijector(np.vstack((x, np.zeros(len(x)))).T.astype(np.float32))
# dir_2 = flow_callback.Y2X_bijector(np.vstack((np.zeros(len(x)), x)).T.astype(np.float32))
# #dir_3 = flow_callback.Z2X_bijector(np.vstack((np.zeros(len(x)), -x)).T.astype(np.float32))
#
# g = plots.get_subplot_plotter()
# g.triangle_plot([chain, flow_chain], params=param_names, filled=False)
# ax = g.subplots[1, 0]
# ax.plot(dir_1[:, 0], dir_1[:, 1])
# ax.plot(dir_2[:, 0], dir_2[:, 1])
# plt.figure()
# plt.plot(dir_1[:, 0], dir_1[:, 1])
# plt.plot(dir_2[:, 0], dir_2[:, 1])
