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
#flow.cop
from scipy import optimize
from scipy.integrate import simps

import synthetic_probability
import importlib
importlib.reload(synthetic_probability)

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
#flow_P = mcmc_tension.DiffFlowCallback(chain, param_names=param_names, feedback=1, learning_rate=0.01)
flow_P = synthetic_probability.DiffFlowCallback(chain, param_names=param_names, feedback=1, learning_rate=0.01)


# train:
from tensorflow.keras.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau()]

batch_size = 8192
epochs = 40
steps_per_epoch = 128

flow_P.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

# plot learned distribution from samples:
N = 10000
X_sample = np.array(flow_P.sample(N))
Y_sample = np.array(flow_P.dist_gaussian_approx.sample(N))
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


omegam = np.linspace(.0, .8, 200)
sigma8 = np.linspace(.2, 1.8, 200)

x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)

# using the method implemented in dist_learned:
log_P = flow_P.log_probability(np.array([X, Y], dtype=np.float32).T)
log_P = np.array(log_P).T
P = np.exp(log_P)
P = P / simps(simps(P, y), x)

# plot:
levels = [utilities.from_sigma_to_confidence(i) for i in range(5, 1, -1)]
plt.contour(X, Y, P, get_levels(P, x, y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels])
density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='--', colors=['red' for i in levels])

###############################################################################
# determinant of the Jacobian:
###############################################################################

omegam = np.linspace(.1, .5, 100)
sigma8 = np.linspace(.6, 1.2, 100)

x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)

# compute log det Jacobian:
log_det = flow_P.log_det_metric(np.array([X, Y], dtype=np.float32).T)
log_det = np.array(log_det).T

pc = plt.pcolormesh(X, Y, log_det, linewidth=0, rasterized=True, shading='auto', cmap='RdBu')
colorbar = plt.colorbar(pc)
density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='--', colors=['red' for i in levels])
plt.xlim([np.amin(omegam), np.amax(omegam)])
plt.ylim([np.amin(sigma8), np.amax(sigma8)])

###############################################################################
# find maximum posterior:
###############################################################################

# MAP:
result = flow_P.MAP_finder(disp=True)
print(result)
maximum_posterior = result.x
# find where the MAP goes:
map_image = flow_P.Z2X_bijector.inverse(np.array(maximum_posterior, dtype=np.float32))
print(maximum_posterior, np.array(map_image))

# mean:
mean = chain.getMeans([chain.index[name] for name in ['omegam', 'sigma8']])
mean_image = flow_P.Z2X_bijector.inverse(np.array(mean, dtype=np.float32))
print(mean, np.array(mean_image))

# plot:
levels = [utilities.from_sigma_to_confidence(i) for i in range(3, 1, -1)]
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
covariance_metric = flow_P.metric(np.array([mean]).astype(np.float32))[0]
fisher_metric = flow_P.inverse_metric(np.array([mean]).astype(np.float32))[0]

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
mode = 0
plt.quiver(mean[0],mean[1],eigv[0,mode], eigv[1,mode], angles = 'xy')
mode = 1
plt.quiver(mean[0],mean[1],eigv[0,mode], eigv[1,mode], angles = 'xy')

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
    geodesics.append(flow_P.map_to_original_coord(geo.T))

# geodesics aligned with abstract coordinate axes:
r = np.linspace(-20.0, 20.0, 1000)

t = 0.0
geo = np.array([map_image[0] + r*np.cos(t),
                map_image[1] + r*np.sin(t)], dtype=np.float32)
geo_1 = flow_P.map_to_original_coord(geo.T)

t = np.pi/2.
geo = np.array([map_image[0] + r*np.cos(t),
                map_image[1] + r*np.sin(t)], dtype=np.float32)
geo_2 = flow_P.map_to_original_coord(geo.T)

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

import matplotlib

r = np.linspace(0.0, 1000.0, 1000)
theta = np.linspace(0.0, 2.0*np.pi, 100)

# copmpute PCA:
eig, eigv = np.linalg.eigh(cov_samples)

# compute geodesics:
geodesics = []
for t in theta:
    geo = np.array([map_image[0] + r*np.cos(t),
                    map_image[1] + r*np.sin(t)], dtype=np.float32)
    geodesics.append(flow_P.map_to_original_coord(geo.T))

# geodesics aligned with abstract coordinate axes:
r = np.linspace(-1000.0, 1000.0, 1000)

t = 0.0
geo = np.array([map_image[0] + r*np.cos(t),
                map_image[1] + r*np.sin(t)], dtype=np.float32)
geo_1 = flow_P.map_to_original_coord(geo.T)

t = np.pi/2.
geo = np.array([map_image[0] + r*np.cos(t),
                map_image[1] + r*np.sin(t)], dtype=np.float32)
geo_2 = flow_P.map_to_original_coord(geo.T)

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

plt.xlim([-100, 100.0])
plt.ylim([-100, 100.0])
plt.legend()

###############################################################################
# local eigenvalues of the metric:
###############################################################################

omegam = np.linspace(.15, .4, 20)
sigma8 = np.linspace(.6, 1.2, 20)

x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)
grid = np.array([X, Y])
points = grid.reshape(2,-1).T
coords = points.astype(np.float32)

local_metric = flow_P.metric(coords)

PCA_eig, PCA_eigv = np.linalg.eigh(local_metric)

idx = np.argsort(PCA_eig, axis = 1)[0][::-1]
PCA_eig = PCA_eig[:,idx]
PCA_eigv = PCA_eigv[:,:,idx]

# Plot
mode = 0
plt.quiver(coords[:,0], coords[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:, 1,mode], color = 'red', angles = 'xy')
mode = 1
plt.quiver(coords[:,0], coords[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:,1, mode], color = 'cadetblue', angles = 'xy')

density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels], zorder=0)
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

eig, eigv = np.linalg.eigh(cov_samples)
mode = 0
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')
mode = 1
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')

plt.xlim([np.amin(omegam), np.amax(omegam)])
plt.ylim([np.amin(sigma8), np.amax(sigma8)])

###############################################################################
# Compute the Levi-Civita connection
###############################################################################


coord = np.array([mean]).astype(np.float32)

inv_metric = flow_P.inverse_metric(coord)
metric_derivative = flow_P.coord_metric_derivative(coord)
term1 = tf.einsum("...ijk -> ...jik", metric_derivative)
term2 = tf.einsum("...ijk -> ...kij", metric_derivative)

0.5*tf.einsum("...ij, ...jkl -> ...ikl", inv_metric, term1 + term2 - metric_derivative)


@tf.function(experimental_relax_shapes=True)
def ode(y, yprime):
    yprimeprime = -tf.einsum("...ijk, ...j, ...k -> ...i", flow_P.levi_civita_connection(np.array([y])), np.array([yprime]), np.array([yprime]))
    return tf.concat([yprime, yprimeprime], axis=0)

y_init = mean.astype(np.float32)
covariance_metric = flow_P.metric(np.array([y_init]).astype(np.float32))[0]
_, eigv = np.linalg.eigh(covariance_metric)
yprime_init = eigv[:, 0]
y0 = tf.concat([y_init, eigv[:, 0]], axis=0)
print(np.shape(y0))
solution_times = tf.linspace(0,1,100)


flow_P.levi_civita_connection(np.array([y_init]))


flow_P.levi_civita_connection([y_init])

-tf.einsum("...ijk, ...j, ...k -> ...i", flow_P.levi_civita_connection(y_init), yprime_init, yprime_init)

results = tfp.math.ode.BDF().solve(ode, initial_time=0., initial_state=y0,
                                   solution_times=solution_times)







###############################################################################
# Hession playground: function method
###############################################################################


flow_P.coord_metric_derivative(np.array([mean]).astype(np.float32))


timeit flow_P.coord_metric_derivative(np.array([mean]).astype(np.float32))

timeit flow_P.coord_metric_derivative(np.array([mean]).astype(np.float32))

###############################################################################
# Hession playground: function method
###############################################################################
omegam = np.linspace(.15, .4, 5)
sigma8 = np.linspace(.6, 1.2, 5)
coords = flow_P.coords_transformed(omegam, sigma8, flow_P.Z2X_bijector.inverse)

np.__version__ # needs to be 1.19.2 or earlier
import time as time
T1 = time.time()
h = flow_P.Hessian(coords, flow_P.Z2X_bijector)
T2 = time.time() - T1
print(h)
print(T2)

###############################################################################
# Hession playground: jacobian method
###############################################################################
omegam = np.linspace(.15, .4, 5)
sigma8 = np.linspace(.6, 1.2, 5)
coords = flow_P.coords_transformed(omegam, sigma8, flow_P.Z2X_bijector.inverse)

np.__version__ # needs to be 1.19.2 or earlier
import time as time
T1 = time.time()
coords_tf = tf.constant(coords.astype(np.float32))
#coords_tf2 = tf.Variable(coords.astype(np.float32))
delta = tf.Variable([0.0,0.0])
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t2:
    t2.watch(delta)
    with  tf.GradientTape(watch_accessed_variables=False, persistent=True) as t1:
        t1.watch(delta)
        #f = flow_P.Z2X_bijector(coords_tf+delta)
        f = (coords_tf+delta)**2 # Checking that Hessian gives expected
    g = t1.jacobian(f,delta)
print(coords_tf[0])
print(g[0,:])
h = t2.jacobian(g,delta)
T2 = time.time() - T1
print(h[0])
print(np.shape(coords_tf),'g', np.shape(g),'h',(np.shape(h)))
print(T2)

###############################################################################
# Hession playground: batch jacobian method (around the same time)
###############################################################################

T1 = time.time()
coords_tf = tf.constant(coords.astype(np.float32))
coords_tf2 = tf.Variable(coords.astype(np.float32))

delta = tf.Variable([0.0,0.0])
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t2:
    t2.watch(coords_tf)
    with  tf.GradientTape(watch_accessed_variables=False, persistent=True) as t1:
        t1.watch(delta)
        f = flow_P.Z2X_bijector(coords_tf+delta)

    g = t1.jacobian(f,delta)

print(g)
h = t2.batch_jacobian(g,coords_tf)
T2 = time.time() - T1
print(h)
print(np.shape(coords_tf),'g', np.shape(g),'h',(np.shape(h)))
print(T2)

###############################################################################
# Hession playground: for loop jacobian method to check
###############################################################################

omegam = np.linspace(.15, .4, 5)
sigma8 = np.linspace(.6, 1.2, 5)


for om in omegam:
    for sig in sigma8:

        P1 = np.array([om, sig])
        P1_prime = np.array(flow_P.Z2X_bijector.inverse(P1.astype(np.float32)))
        x = tf.constant(P1_prime.astype(np.float32))
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t2:
            t2.watch(x)
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t1:
                t1.watch(x)
                y = flow_P.Z2X_bijector(x)
            grad = t1.jacobian(y, x)
        hess = t2.jacobian(grad, x)
        print(hess)



###############################################################################
# Determinant (2 methods):
###############################################################################

det, det_met = flow_P.det_metric(omegam, sigma8, flow_P.Z2X_bijector)

###############################################################################
# TESTING NEW CLASS FUNCTIONS:
###############################################################################
omegam = np.linspace(.15, .4, 5)
sigma8 = np.linspace(.6, 1.2, 5)
x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)
grid = np.array([X,Y])
coords = (grid.reshape(2,-1).T).astype(np.float32)

print(flow_P.f)
jac = flow_P.direct_jacobian(coords)
print((jac))
print(tf.transpose(jac))
metric_method = flow_P.metric()
print(metric_method)

###############################################################################
# Metric (class functions method):
###############################################################################
omegam = np.linspace(.15, .4, 20)
sigma8 = np.linspace(.6, 1.2, 20)
#importlib.reload(flow_copy)
x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)
grid = np.array([X,Y])
points = grid.reshape(2,-1).T
P1 = points
#coords = flow_P.coords_transformed(omegam, sigma8, flow_P.Z2X_bijector.inverse)
#metric_method = flow_P.Metric(omegam, sigma8, flow_P.Z2X_bijector)

print(flow_P.f)
jac = flow_P.Jacobian()
print((jac))
print(tf.transpose(jac))
metric_method = flow_P.metric()
print(metric_method)

PCA_eig, PCA_eigv = np.linalg.eigh(metric_method)

idx = np.argsort(PCA_eig, axis = 1)[0][::-1]
PCA_eig = PCA_eig[:,idx]
PCA_eigv = PCA_eigv[:,:,idx]

# Plot
plt.figure(figsize = (10,10))
mode = 0
plt.quiver(P1[:,0], P1[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:, 1,mode], color = 'red', angles = 'xy')
mode = 1
plt.quiver(P1[:,0], P1[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:,1, mode], color = 'cadetblue', angles = 'xy')

density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels], zorder=0)
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

eig, eigv = np.linalg.eigh(cov_samples)
mode = 0
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')
mode = 1
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')

plt.xlim([np.amin(omegam), np.amax(omegam)])
plt.ylim([np.amin(sigma8), np.amax(sigma8)])

###############################################################################
# Metric (faster method):
###############################################################################
omegam = np.linspace(.15, .4, 20)
sigma8 = np.linspace(.6, 1.2, 20)

x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)
grid = np.array([X,Y])
points = grid.reshape(2,-1).T
P1 = points
P1_prime = np.array((flow_P.Z2X_bijector.inverse)(P1.astype(np.float32)))
#P1_prime = np.array((X2Z_bijector(P1.astype(np.float32))))

#x = tf.Variable(P1_prime.astype(np.float32)) #variable or constant doesn't seem to matter
x = (P1_prime)
delta = tf.Variable([0.0,0.0])
with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
    g.watch(delta)
    y = flow_P.Z2X_bijector(x+delta)
jac = g.jacobian(y, delta)
print(jac)

jac = np.array(jac)
jac_T = np.transpose(jac, (0,2,1))

met = np.matmul(jac,jac_T)

# metric[-1] from faster method:
#[[0.06253082 0.07325305]
#[0.07325305 0.0869597 ]]
# metric[-1] from loop method:
#[[0.06253086 0.07325337]
#[0.07325337 0.0869604 ]]



print(met[:5])
print(metric_method[:5])
PCA_eig, PCA_eigv = np.linalg.eigh(met)

# sort:
idx = np.argsort(PCA_eig, axis = 1)[0][::-1]
PCA_eig = PCA_eig[:,idx]
PCA_eigv = PCA_eigv[:,:,idx]



# Plot
plt.figure(figsize = (10,10))
mode = 0
plt.quiver(P1[:,0], P1[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:, 1,mode], color = 'red', angles = 'xy')
mode = 1
plt.quiver(P1[:,0], P1[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:,1, mode], color = 'cadetblue', angles = 'xy')

density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels], zorder=0)
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

eig, eigv = np.linalg.eigh(cov_samples)
mode = 0
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')
mode = 1
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')

plt.xlim([np.amin(omegam), np.amax(omegam)])
plt.ylim([np.amin(sigma8), np.amax(sigma8)])

plt.savefig('test.pdf')

###############################################################################
# Metric (loop method):
###############################################################################
omegam = np.linspace(.15, .4, 10)
sigma8 = np.linspace(.6, 1.2, 10)

x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)

for om in omegam:
    for sig in sigma8:

        P1 = np.array([om, sig])
        P1_prime = np.array(flow_P.Z2X_bijector.inverse(P1.astype(np.float32)))
        x = tf.constant(P1_prime.astype(np.float32))
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as g:
            g.watch(x)
            y = flow_P.Z2X_bijector(x)
        jac = g.jacobian(y, x)
        mu = np.identity(2)
        metric = np.dot(np.array(jac), np.dot(mu, np.array(jac).T))
        PCA_eig, PCA_eigv = np.linalg.eigh(metric)
        # sort:
        idx = np.argsort(PCA_eig)[::-1]
        PCA_eig = PCA_eig[idx]
        PCA_eigv = PCA_eigv[:, idx]
        mode = 0
        plt.quiver(P1[0], P1[1], PCA_eigv[0, mode], PCA_eigv[1, mode], color = 'red', angles = 'xy')
        mode = 1
        plt.quiver(P1[0], P1[1], PCA_eigv[0, mode], PCA_eigv[1, mode], color = 'cadetblue', angles = 'xy')

density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels], zorder=0)
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')

eig, eigv = np.linalg.eigh(cov_samples)
mode = 0
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')
mode = 1
plt.plot(maximum_posterior[0] + r*eigv[0, mode], maximum_posterior[1] + r*eigv[1, mode], ls='-', color='k')

plt.xlim([np.amin(omegam), np.amax(omegam)])
plt.ylim([np.amin(sigma8), np.amax(sigma8)])

print(metric)
print(jac)
print(PCA_eig)
print(PCA_eigv)

plt.savefig('test.pdf')

    def grid_coords_transformed(self, x_array, y_array, bijector_inv):
        """
        """
        X, Y = np.meshgrid(x_array, y_array)
        grid = np.array([X, Y])
        coords0 = grid.reshape(2, -1).T
        coords = np.array((bijector_inv)(coords0.astype(np.float32)))
        #
        return coords
