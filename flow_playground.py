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
from tensiometer.tensiometer import utilities
from tensiometer.tensiometer import gaussian_tension
from tensiometer.tensiometer import mcmc_tension
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions


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
flow_callback = mcmc_tension.DiffFlowCallback(chain, param_names=param_names, feedback=1, learning_rate=0.01)

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
flow_chain = MCSamples(samples=X_sample, names=param_names, label='Learned distribution')

g = plots.get_subplot_plotter()
g.triangle_plot([chain, flow_chain], params=param_names, filled=False)

# slice two dimensions:
x = np.linspace(0.0, 2.0, 100)
dir_1 = flow_callback.Z2X_bijector(np.vstack((x, np.zeros(len(x)))).T.astype(np.float32))
dir_2 = flow_callback.Z2X_bijector(np.vstack((np.zeros(len(x)), x)).T.astype(np.float32))
#dir_3 = flow_callback.Z2X_bijector(np.vstack((-x, np.zeros(len(x)))).T.astype(np.float32))

g = plots.get_subplot_plotter()
g.triangle_plot([chain, flow_chain], params=param_names, filled=False)
ax = g.subplots[1, 0]
ax.plot(dir_1[:, 0], dir_1[:, 1])
ax.plot(dir_2[:, 0], dir_2[:, 1])
#ax.plot(dir_3[:, 0], dir_3[:, 1])

# Trying to get inverted plot:

X2Z_bijector = tfb.Invert(flow_callback.Z2X_bijector)
num_params = flow_callback.num_params
dist_learned_inverted = tfd.TransformedDistribution(distribution=flow_callback.dist_learned, bijector=X2Z_bijector)

#X_sample1 = np.array(flow_callback.dist_transformed.sample(N))
Z_sample = np.array(dist_learned_inverted.sample(N))

flow_chain1 = MCSamples(samples=Z_sample, names=['Z1', 'Z2'], label='Transformed distribution')

g = plots.get_subplot_plotter()
g.triangle_plot([flow_chain1], params=['Z1', 'Z2'], filled=False)

# slice two dimensions:
x = np.linspace(0.0, 2.0, 100)
dir_1 = (np.vstack((x, np.zeros(len(x)))).T.astype(np.float32))
dir_2 = (np.vstack((np.zeros(len(x)), x)).T.astype(np.float32))
ax = g.subplots[1, 0]
ax.plot(dir_1[:, 0], dir_1[:, 1])
ax.plot(dir_2[:, 0], dir_2[:, 1])

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
