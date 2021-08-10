
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

# find maximum posterior:

# MAP:
result = flow_P.MAP_finder(disp=True)
maximum_posterior = result.x
# find where the MAP goes:
map_image = flow_P.map_to_abstract_coord(np.array(maximum_posterior, dtype=np.float32))
# mean:
mean = chain.getMeans([chain.index[name] for name in ['omegam', 'sigma8']])
mean_image = flow_P.map_to_abstract_coord(np.array(mean, dtype=np.float32))
# covariance from samples
cov_samples = chain.cov(pars=['omegam', 'sigma8'])
# probability levels:
levels = [utilities.from_sigma_to_confidence(i) for i in range(3, 0, -1)]

###############################################################################
# local eigenvalues of the metric:
###############################################################################

r = np.linspace(-20.0, 20.0, 1000)
omegam = np.linspace(.15, .4, 20)
sigma8 = np.linspace(.6, 1.2, 20)

x, y = omegam, sigma8
X, Y = np.meshgrid(x, y)
coords = np.array([X, Y], dtype = np.float32).reshape(2,-1).T

local_metric = flow_P.metric(coords)

PCA_eig, PCA_eigv = np.linalg.eigh(local_metric)
idx = np.argsort(PCA_eig, axis = 1)[0]#[::-1]
PCA_eig = PCA_eig[:,idx]
PCA_eigv = PCA_eigv[:,:,idx]

# Plot
plt.figure(figsize = (8,8))
mode = 0
plt.quiver(coords[:,0], coords[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:, 1,mode], color = 'red', angles = 'xy', label = 'First mode')
mode = 1
plt.quiver(coords[:,0], coords[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:,1, mode], color = 'cadetblue', angles = 'xy', label = 'Second mode')

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
plt.legend()

###############################################################################
# Geodesics
###############################################################################

# add plot of geodesics that start from map and go around, as in the previous case
@tf.function()
def levi_civita_connection(coord):
    """
    Compute the Levi-Civita connection
    """
    inv_metric = flow_P.inverse_metric(coord)
    metric_derivative = flow_P.coord_metric_derivative(coord)
    term_1 = tf.einsum("...ij, ...kjl -> ...ikl", inv_metric, metric_derivative)
    term_2 = tf.einsum("...ij, ...ljk -> ...ikl", inv_metric, metric_derivative)
    term_3 = tf.einsum("...ij, ...klj -> ...ikl", inv_metric, metric_derivative)
    #
    return 0.5*(term_1 + term_2 - term_3)


coord = np.array([mean]).astype(np.float32)
coord

flow_P.metric(coord)
flow_P.direct_jacobian(coord)

levi_civita_connection(coord)
flow_P.coord_metric_derivative(coord)

@tf.function()
def ode(t, y, n):
    # unpack position and velocity:
    pos = y[:n]
    vel = y[n:]
    # compute geodesic equation:
    acc = -tf.einsum("...ijk, ...j, ...k -> ...i", levi_civita_connection(tf.convert_to_tensor([pos])), tf.convert_to_tensor([vel]), tf.convert_to_tensor([vel]))
    #
    return tf.concat([vel, acc[0]], axis=0)

y_init = maximum_posterior.astype(np.float32)
covariance_metric = flow_P.metric(np.array([y_init]).astype(np.float32))[0]
eig, eigv = np.linalg.eigh(covariance_metric)
yprime_init = eigv[:, 1]/np.sqrt(eig[1])
y0 = tf.concat([y_init, yprime_init], axis=0)
solution_times = tf.linspace(0., 0.1, 20)

print('Norm of initial velocity', np.dot(np.dot(yprime_init, covariance_metric), yprime_init))

results = tfp.math.ode.DormandPrince().solve(ode, initial_time=0., initial_state=y0, solution_times=solution_times, constants={'n': 2})
#results = tfp.math.ode.BDF().solve(ode, initial_time=0., initial_state=y0, solution_times=solution_times, constants={'n': 2})

# check conservation of velocity modulus:
temp_metric = flow_P.metric(np.array([results.states[:, 0], results.states[:, 1]]).T.astype(np.float32)).numpy()
velocity = np.array([results.states[:, 2], results.states[:, 3]]).T

res = []
for g, v in zip(temp_metric, velocity):
    res.append(np.dot(np.dot(v, g), v))
res = np.array(res)
plt.plot(results.times.numpy(), res)


print(yprime_init)
plt.plot(results.states[:, 0], results.states[:, 1])
plt.quiver(results.states[:,0], results.states[:,1], results.states[:, 2], results.states[:, 3], color = 'red', angles = 'xy')
density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels], zorder=0)
#plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')
#plt.xlim(.25,.35)
#plt.ylim(.8,1)
plt.savefig('test.pdf')

###############################################################################
# PCA flow
###############################################################################

# add plot of PCA flow from many points in parameter space

@tf.function()
def ode(t, y, n):
    # compute metric:
    metric = flow_P.metric(tf.convert_to_tensor([y]))
    # compute eigenvalues:
    _, eigv = tf.linalg.eigh(metric[0])
    #
    return eigv[:, n]

y_init = mean.astype(np.float32)
y = y_init
y0 = y_init
solution_times = tf.linspace(0, 1, 100)

results = tfp.math.ode.DormandPrince().solve(ode, initial_time=0., initial_state=y0, solution_times=solution_times, constants={'n': 0})

plt.plot(results.states[:, 0], results.states[:, 1])

# Plot
mode = 0
plt.quiver(coords[:,0], coords[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:, 1,mode], color = 'red', angles = 'xy')
mode = 1
plt.quiver(coords[:,0], coords[:,1], PCA_eigv[:, 0,mode], PCA_eigv[:,1, mode], color = 'cadetblue', angles = 'xy')

density = chain.get2DDensity('omegam', 'sigma8', normalized=True)
_X, _Y = np.meshgrid(density.x, density.y)
plt.contour(_X, _Y, density.P, get_levels(density.P, density.x, density.y, levels), linewidths=1., linestyles='-', colors=['k' for i in levels], zorder=0)
plt.scatter(maximum_posterior[0], maximum_posterior[1], color='k')




# can we clean up the following?


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



import analyze_2d_example
import importlib
importlib.reload(analyze_2d_example)
analyze_2d_example.run_example_2d(chain, prior_chain, ['omegam','sigma8'], [[.1,.5],[.55,1.25]], outroot = '/Users/TaraD/Downloads/')
