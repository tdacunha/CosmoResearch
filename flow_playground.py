# Show plots inline, and load main getdist plot module and samples class
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
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions

#import flow_nonunit
import flow_copy

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

# plot learned distribution:
N = 10000
X_sample = np.array(flow_callback.dist_learned.sample(N))
Y_sample = np.array(flow_callback.dist_gaussian_approx.sample(N))
flow_chain = MCSamples(samples=X_sample, names=param_names, label='Learned distribution')
Y_chain = MCSamples(samples=Y_sample, names=param_names, label='Transformed distribution')


g = plots.get_subplot_plotter()
g.triangle_plot([chain, flow_chain, Y_chain], params=param_names, filled=False)

# slice two dimensions:
x = np.linspace(0.0, 2.0, 100)

dir_1 = flow_callback.Z2X_bijector(np.vstack((x, np.zeros(len(x)))).T.astype(np.float32))
dir_2 = flow_callback.Z2X_bijector(np.vstack((np.zeros(len(x)), x)).T.astype(np.float32))

g = plots.get_subplot_plotter()
g.triangle_plot([chain, flow_chain], params=param_names, filled=False)
ax = g.subplots[1, 0]
ax.plot(dir_1[:, 0], dir_1[:, 1])
ax.plot(dir_2[:, 0], dir_2[:, 1])

# Experimenting with metric field plot:
#from tensorflow_probability.python.math import gradient
import tensorflow as tf
#tf.function(experimental_relax_shapes=True)
#@tf.function(experimental_relax_shapes=True)
X2Z_bijector = tfb.Invert(flow_callback.Z2X_bijector)
Y2Z_bijector = tfb.Invert(flow_callback.Z2Y_bijector)

p = plots.get_subplot_plotter(subplot_size = 5)
p.plot_2d([chain, flow_chain], param1=param_names[0], param2 = param_names[1], filled=False)
ax = p.subplots[0,0]

for om in np.linspace(.1,.5,10):
    for sig in np.linspace(.6,1.2,10):
        P1 = np.array([om,sig])
        P1_prime = np.array(X2Z_bijector(P1.astype(np.float32)))
        #P1_primeprime = np.array(flow_callback.Z2Y_bijector(P1_prime.astype(np.float32)))
        #print(P1,P1_primeprime)
        #_,grads = gradient.value_and_gradient(flow_callback.Z2X_bijector, P1_prime)
        #_1,jac = gradient.value_and_batch_jacobian(flow_callback.Z2X_bijector(x), x)

        x = tf.constant(P1_prime.astype(np.float32)) # or tf.constant, Variable but Variable doesn;t work with tf.function
        #delta = tf.Variable([0.0,0.0])
        with tf.GradientTape() as g: #persistent true doesn't seem to affect rn
        #g = tf.GradientTape(persistent=True)

            g.watch(x)

            y = flow_callback.Z2X_bijector(x)
        jac = g.jacobian(y,x)
        #print(np.array(jac))
        #print(type(jac))
        mu = np.identity(2)
        metric = np.matmul((np.array(jac)).T,np.matmul(mu,(np.array(jac))))
        #print(metric)
        PCA_eig,PCA_eigv = np.linalg.eig(metric)
        print(PCA_eigv)
        #print(PCA_eigv)
        #print(PCA_eig) #second value is first mode?
        ax.quiver(P1[0],P1[1],PCA_eigv[0,1],PCA_eigv[1,1], color = 'cadetblue')
plt.show()








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
