"""

TODO:

- implement MAP finder
- implement base geometry methods
- clean up used modules
- add method to save to file (and cache) the learned distributrion

For testing purposes:

import getdist
chains_dir = './tensiometer/test_chains/'
settings = {'ignore_rows':0, 'smooth_scale_1D':0.3, 'smooth_scale_2D':0.3}
chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'DES', no_cache=True, settings=settings)
param_names = ['omegam', 'sigma8']
from tensorflow.keras.callbacks import Callback
self = Callback()
"""

###############################################################################
# initial imports and set-up:

import os
import time
import gc
import copy
from numba import jit
import numpy as np
import getdist.chains as gchains
gchains.print_load_details = False
from getdist import MCSamples, WeightedSamples
import scipy
from scipy.linalg import sqrtm
from scipy.integrate import simps
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution
import scipy.stats
import pickle
from collections.abc import Iterable
from matplotlib import pyplot as plt

#from .. import utilities as utils
#from .. import gaussian_tension
from tensiometer import utilities as utils #edited by tara
from tensiometer import gaussian_tension #edited by tara

try:
    import tensorflow as tf
    import tensorflow_probability as tfp
    tfb = tfp.bijectors
    tfd = tfp.distributions
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input
    from tensorflow.keras.callbacks import Callback
    HAS_FLOW = True
except Exception as e:
    print("Could not import tensorflow or tensorflow_probability: ", e)
    Callback = object
    HAS_FLOW = False

try:
    from IPython.display import clear_output, set_matplotlib_formats
except ModuleNotFoundError:
    pass

###############################################################################
# helper class to build a masked-autoregressive flow:


class SimpleMAF(object):
    """
    A class to implement a simple Masked AutoRegressive Flow (MAF) using the implementation :class:`tfp.bijectors.AutoregressiveNetwork` from from `Tensorflow Probability <https://www.tensorflow.org/probability/>`_. Additionally, this class provides utilities to load/save models, including random permutations.

    :param num_params: number of parameters, ie the dimension of the space of which the bijector is defined.
    :type num_params: int
    :param n_maf: number of MAFs to stack. Defaults to None, in which case it is set to `2*num_params`.
    :type n_maf: int, optional
    :param hidden_units: a list of the number of nodes per hidden layers. Defaults to None, in which case it is set to `[num_params*2]*2`.
    :type hidden_units: list, optional
    :param permutations: whether to use shuffle dimensions between stacked MAFs, defaults to True.
    :type permutations: bool, optional
    :param activation: activation function to use in all layers, defaults to :func:`tf.math.asinh`.
    :type activation: optional
    :param kernel_initializer: kernel initializer, defaults to 'glorot_uniform'.
    :type kernel_initializer: str, optional
    :param feedback: print the model architecture, defaults to 0.
    :type feedback: int, optional
    :reference: George Papamakarios, Theo Pavlakou, Iain Murray (2017). Masked Autoregressive Flow for Density Estimation. `arXiv:1705.07057 <https://arxiv.org/abs/1705.07057>`_
    """

    def __init__(self, num_params, n_maf=None, hidden_units=None, permutations=True, activation=tf.math.asinh, kernel_initializer='glorot_uniform', feedback=0, **kwargs):
        if n_maf is None:
            n_maf = 2*num_params
        event_shape = (num_params,)

        if hidden_units is None:
            hidden_units = [num_params*2]*2

        if permutations is None:
            _permutations = False
        elif isinstance(permutations, Iterable):
            assert len(permutations) == n_maf
            _permutations = permutations
        elif isinstance(permutations, bool):
            if permutations:
                _permutations = [np.random.permutation(num_params) for _ in range(n_maf)]
            else:
                _permutations = False

        self.permutations = _permutations

        # Build transformed distribution
        bijectors = []
        for i in range(n_maf):
            if _permutations:
                bijectors.append(tfb.Permute(_permutations[i].astype(np.int32)))
            made = tfb.AutoregressiveNetwork(params=2, event_shape=event_shape, hidden_units=hidden_units, activation=activation, kernel_initializer=kernel_initializer, **kwargs)
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=made))

        self.bijector = tfb.Chain(bijectors)

        if feedback > 0:
            print("Building MAF")
            print("    - number of MAFs:", n_maf)
            print("    - activation:", activation)
            print("    - hidden_units:", hidden_units)

    def save(self, path):
        """
        Save a `SimpleMAF` object.

        :param path: path of the directory where to save.
        :type path: str
        """
        checkpoint = tf.train.Checkpoint(bijector=self.bijector)
        checkpoint.write(path)
        pickle.dump(self.permutations, open(path+'_permutations.pickle', 'wb'))

    @classmethod
    def load(cls, num_params, path, **kwargs):
        """
        Load a saved `SimpleMAF` object. The number of parameters and all other keyword arguments (except for `permutations`) must be included as the MAF is first created with random weights and then these weights are restored.

        :param num_params: number of parameters, ie the dimension of the space of which the bijector is defined.
        :type num_params: int
        :param path: path of the directory from which to load.
        :type path: str
        :return: a :class:`~.SimpleMAF`.
        """
        permutations = pickle.load(open(path+'_permutations.pickle', 'rb'))
        maf = SimpleMAF(num_params=num_params, permutations=permutations, **kwargs)
        checkpoint = tf.train.Checkpoint(bijector=maf.bijector)
        checkpoint.read(path)
        return maf

###############################################################################
# main class to compute NF-based tension:

class DiffFlowCallback(Callback):
    """
    A class to compute the normalizing flow interpolation of a probability density given the samples.

    A normalizing flow is trained to approximate the distribution and then used to numerically evaluate the probablity of a parameter shift (see REF). To do so, it defines a bijective mapping that is optimized to gaussianize the difference chain samples. This mapping is performed in two steps, using the gaussian approximation as pre-whitening. The notations used in the code are:

    * `X` designates samples in the original parameter difference space;
    * `Y` designates samples in the gaussian approximation space, `Y` is obtained by shifting and scaling `X` by its mean and covariance (like a PCA);
    * `Z` designates samples in the gaussianized space, connected to `Y` with a normalizing flow denoted `Z2Y_bijector`.

    The user may provide the `Z2Y_bijector` as a :class:`~tfp.bijectors.Bijector` object from `Tensorflow Probability <https://www.tensorflow.org/probability/>`_ or make use of the utility class :class:`~.SimpleMAF` to instantiate a Masked Autoregressive Flow (with `Z2Y_bijector='MAF'`).

    This class derives from :class:`~tf.keras.callbacks.Callback` from Keras, which allows for visualization during training. The normalizing flows (X->Y->Z) are implemented as :class:`~tfp.bijectors.Bijector` objects and encapsulated in a Keras :class:`~tf.keras.Model`.

    Here is an example:

    .. code-block:: python

        # Initialize the flow and model
        diff_flow_callback = DiffFlowCallback(chain, Z2Y_bijector='MAF')
        # Train the model
        diff_flow_callback.train()
        # Compute the shift probability and confidence interval
        p, p_low, p_high = diff_flow_callback.estimate_shift_significance()

    :param chain: input parameter difference chain.
    :type chain: :class:`~getdist.mcsamples.MCSamples`
    :param param_names: parameter names of the parameters to be used
        in the calculation. By default all running parameters.
    :type param_names: list, optional
    :param Z2Y_bijector: either a :class:`~tfp.bijectors.Bijector` object
        representing the mapping from `Z` to `Y`, or 'MAF' to instantiate a :class:`~.SimpleMAF`, defaults to 'MAF'.
    :type Z2Y_bijector: optional
    :param pregauss_bijector: not implemented yet, defaults to None.
    :type pregauss_bijector: optional
    :param learning_rate: initial learning rate, defaults to 1e-3.
    :type learning_rate: float, optional
    :param feedback: feedback level, defaults to 1.
    :type feedback: int, optional
    :param validation_split: fraction of samples to use for the validation sample, defaults to 0.1
    :type validation_split: float, optional
    :raises NotImplementedError: if `pregauss_bijector` is not None.
    :reference: George Papamakarios, Theo Pavlakou, Iain Murray (2017). Masked Autoregressive Flow for Density Estimation. `arXiv:1705.07057 <https://arxiv.org/abs/1705.07057>`_
    """

    """
    For testing purposes:
    feedback = 1
    validation_split=0.1
    Z2Y_bijector='MAF'
    pregauss_bijector=None
    learning_rate=1e-3
    feedback=1
    validation_split=0.1
    kwargs={}
    """

    def __init__(self, chain, param_names=None, param_ranges=None, Z2Y_bijector='MAF', pregauss_bijector=None, learning_rate=1e-3, feedback=1, validation_split=0.1, **kwargs):

        # read in varaiables:
        self.feedback = feedback

        # Chain
        self._init_chain(chain, param_names=param_names, param_ranges=param_ranges, validation_split=validation_split)

        # Transformed distribution
        self._init_transf_dist(Z2Y_bijector, learning_rate=learning_rate, **kwargs)
        if feedback > 0:
            print("Building flow")
            print("    - trainable parameters:", self.model.count_params())

        # Metrics
        keys = ["loss", "val_loss", "chi2Z_ks", "chi2Z_ks_p"]
        self.log = {_k: [] for _k in keys}

        self.chi2Y = np.sum(self.Y_test**2, axis=1)
        self.chi2Y_ks, self.chi2Y_ks_p = scipy.stats.kstest(self.chi2Y, 'chi2', args=(self.num_params,))

        # Pre-gaussianization
        if pregauss_bijector is not None:
            # The idea is to introduce yet another step of deterministic gaussianization, eg using the prior CDF
            # or double prior (convolved with itself, eg a triangular distribution)
            raise NotImplementedError

        # internal variables:
        self.is_trained = False

        # initialize geometry:
        self._init_geometry()

    def _init_chain(self, chain, param_names=None, param_ranges=None, validation_split=0.1):
        """
        Add documentation
        """
        # initialize param names:
        if param_names is None:
            param_names = chain.getParamNames().getRunningNames()
        else:
            chain_params = chain.getParamNames().list()
            if not np.all([name in chain_params for name in param_names]):
                raise ValueError('Input parameter is not in the chain.\n',
                                 'Input parameters ', param_names, '\n'
                                 'Possible parameters', chain_params)

        # initialize ranges:
        self.parameter_ranges = {}
        for name in param_names:
            # get ranges from user:
            if param_ranges is not None:
                if name not in param_ranges.keys():
                    raise ValueError('Range for parameter ', name, ' is not specified.\n',
                                     'When passing ranges explicitly all parameters have to be included.')
                else:
                    self.parameter_ranges[name] = copy.deepcopy(param_ranges[name])
            # get ranges from MCSamples:
            else:
                temp_range = []
                # lower:
                if name in chain.ranges.lower.keys():
                    temp_range.append(chain.ranges.lower[name])
                else:
                    temp_range.append(np.amin(chain.samples[:, chain.index[name]]))
                # upper:
                if name in chain.ranges.upper.keys():
                    temp_range.append(chain.ranges.upper[name])
                else:
                    temp_range.append(np.amax(chain.samples[:, chain.index[name]]))
                # save:
                self.parameter_ranges[name] = copy.deepcopy(temp_range)

        # indexes:
        ind = [chain.index[name] for name in param_names]
        self.num_params = len(ind)

        # Gaussian approximation (full chain)
        mcsamples_gaussian_approx = gaussian_tension.gaussian_approximation(chain, param_names=param_names)
        self.dist_gaussian_approx = tfd.MultivariateNormalTriL(loc=mcsamples_gaussian_approx.means[0].astype(np.float32), scale_tril=tf.linalg.cholesky(mcsamples_gaussian_approx.covs[0].astype(np.float32)))
        self.Y2X_bijector = self.dist_gaussian_approx.bijector

        # Samples
        # Split training/test
        n = chain.samples.shape[0]
        indices = np.random.permutation(n)
        n_split = int(validation_split*n)
        test_idx, training_idx = indices[:n_split], indices[n_split:]

        # Training
        self.X = chain.samples[training_idx, :][:, ind]
        self.weights = chain.weights[training_idx]
        self.weights *= len(self.weights) / np.sum(self.weights)  # weights normalized to number of samples
        self.has_weights = np.any(self.weights != self.weights[0])
        self.Y = np.array(self.Y2X_bijector.inverse(self.X.astype(np.float32)))
        assert not np.any(np.isnan(self.Y))
        self.num_samples = len(self.X)

        # Test
        self.X_test = chain.samples[test_idx, :][:, ind]
        self.Y_test = np.array(self.Y2X_bijector.inverse(self.X_test.astype(np.float32)))
        self.weights_test = chain.weights[test_idx]
        self.weights_test *= len(self.weights_test) / np.sum(self.weights_test)  # weights normalized to number of samples

        # Training sample generator
        Y_ds = tf.data.Dataset.from_tensor_slices((self.Y.astype(np.float32),                     # input
                                                   np.zeros(self.num_samples, dtype=np.float32),  # output (dummy zero)
                                                   self.weights.astype(np.float32),))             # weights
        Y_ds = Y_ds.prefetch(tf.data.experimental.AUTOTUNE).cache()
        self.Y_ds = Y_ds.shuffle(self.num_samples, reshuffle_each_iteration=True).repeat()

        if self.feedback:
            print("Building training/test samples")
            if self.has_weights:
                print("    - {}/{} training/test samples and non-uniform weights.".format(self.num_samples, self.X_test.shape[0]))
            else:
                print("    - {}/{} training/test samples and uniform weights.".format(self.num_samples, self.X_test.shape[0]))

    def _init_transf_dist(self, Z2Y_bijector, learning_rate=1e-4, **kwargs):
        """
        Add documentation
        """
        # Model
        if Z2Y_bijector == 'MAF':
            self.MAF = SimpleMAF(self.num_params, feedback=self.feedback, **kwargs)
            Z2Y_bijector = self.MAF.bijector
        assert isinstance(Z2Y_bijector, tfp.bijectors.Bijector)

        # Bijector and transformed distribution
        self.Z2Y_bijector = Z2Y_bijector
        self.dist_transformed = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(np.zeros(self.num_params, dtype=np.float32), np.ones(self.num_params, dtype=np.float32)), bijector=Z2Y_bijector)

        # Full bijector
        self.Z2X_bijector = tfb.Chain([self.Y2X_bijector, self.Z2Y_bijector])

        # Full distribution
        self.dist_learned = tfd.TransformedDistribution(distribution=tfd.MultivariateNormalDiag(np.zeros(self.num_params, dtype=np.float32), np.ones(self.num_params, dtype=np.float32)), bijector=self.Z2X_bijector)  # samples from std gaussian mapped to X

        # Construct model
        x_ = Input(shape=(self.num_params,), dtype=tf.float32)
        log_prob_ = self.dist_transformed.log_prob(x_)
        self.model = Model(x_, log_prob_)

        loss = lambda _, log_prob: -log_prob

        self.model.compile(optimizer=tf.optimizers.Adam(learning_rate=learning_rate), loss=loss)


    def train(self, epochs=100, batch_size=None, steps_per_epoch=None, callbacks=[], verbose=1, **kwargs):
        """
        Train the normalizing flow model. Internallay, this runs the fit method of the Keras :class:`~tf.keras.Model`, to which `**kwargs are passed`.

        :param epochs: number of training epochs, defaults to 100.
        :type epochs: int, optional
        :param batch_size: number of samples per batch, defaults to None. If None, the training sample is divided into `steps_per_epoch` batches.
        :type batch_size: int, optional
        :param steps_per_epoch: number of steps per epoch, defaults to None. If None and `batch_size` is also None, then `steps_per_epoch` is set to 100.
        :type steps_per_epoch: int, optional
        :param callbacks: a list of additional Keras callbacks, such as :class:`~tf.keras.callbacks.ReduceLROnPlateau`, defaults to [].
        :type callbacks: list, optional
        :param verbose: verbosity level, defaults to 1.
        :type verbose: int, optional
        :return: A :class:`~tf.keras.callbacks.History` object. Its `history` attribute is a dictionary of training and validation loss values and metrics values at successive epochs: `"shift0_chi2"` is the squared norm of the zero-shift point in the gaussianized space, with the probability-to-exceed and corresponding tension in `"shift0_pval"` and `"shift0_nsigma"`; `"chi2Z_ks"` and `"chi2Z_ks_p"` contain the :math:`D_n` statistic and probability-to-exceed of the Kolmogorov-Smironov test that squared norms of the transformed samples `Z` are :math:`\\chi^2` distributed (with a number of degrees of freedom equal to the number of parameters).
        """
        """
        For testing purposes:
        epochs=100
        batch_size=None
        steps_per_epoch=None
        callbacks=[]
        verbose=1
        kwargs = {}
        """
        # We're trying to loop through the full sample each epoch
        if batch_size is None:
            if steps_per_epoch is None:
                steps_per_epoch = 100
            batch_size = int(self.num_samples/steps_per_epoch)
        else:
            if steps_per_epoch is None:
                steps_per_epoch = int(self.num_samples/batch_size)

        # Run !
        hist = self.model.fit(x=self.Y_ds.batch(batch_size),
                              batch_size=batch_size,
                              epochs=epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data=(self.Y_test, np.zeros(len(self.Y_test), dtype=np.float32), self.weights_test),
                              verbose=verbose,
                              callbacks=[tf.keras.callbacks.TerminateOnNaN(), self]+callbacks,
                              **kwargs)
        # model is now trained:
        self.is_trained = True
        #
        return hist

    ###############################################################################
    # Utility functions:

    @tf.function()
    def sample(self, N):
        """
        Return samples from the synthetic probablity.
        """
        return self.dist_learned.sample(N)

    @tf.function()
    def log_probability(self, coord):
        """
        Returns learned log probability.
        """
        return self.dist_learned.log_prob(coord)

    def MAP_finder(self, **kwargs):
        """
        Function that uses scipy differential evolution to find the global maximum of the synthetic posterior.
        """
        # main call to differential evolution:
        result = differential_evolution(lambda x: -self.dist_learned.log_prob(np.array(x, dtype=np.float32)),
                                        bounds=list(self.parameter_ranges.values()),
                                        **kwargs)
        # cache MAP value:
        if result.success:
            self.MAP_coord = result.x
            self.MAP_logP = -result.fun
        #
        return result

    ###############################################################################
    # Information geometry methods:

    @tf.function()
    def map_to_abstract_coord(self, coord):
        """
        Map from parameter space to abstract space
        """
        return self.Z2X_bijector.inverse(coord)

    @tf.function()
    def map_to_original_coord(self, coord):
        """
        Map from abstract space to parameter space
        """
        return self.Z2X_bijector(coord)

    @tf.function()
    def log_det_metric(self, coord):
        """
        Computes the log determinant of the metric
        """
        log_det = self.Z2X_bijector.inverse_log_det_jacobian(coord, event_ndims=1)
        return 2.*log_det

    @tf.function()
    def direct_jacobian(self, coord):
        """
        Computes the Jacobian of the parameter transformation at one point in (original) parameter space
        """
        abs_coord = self.map_to_abstract_coord(coord)
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(abs_coord)
            f = self.map_to_original_coord(abs_coord)
        return tape.batch_jacobian(f, abs_coord)

    @tf.function()
    def inverse_jacobian(self, coord):
        """
        Computes the inverse Jacobian of the parameter transformation at one point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.map_to_abstract_coord(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def metric(self, coord):
        """
        Computes the metric at a given point or array of points in (original) parameter space
        """
        # compute Jacobian:
        jac = self.inverse_jacobian(coord)
        # take the transpose (we need to calculate the indexes that we want to swap):
        trailing_axes = [-1, -2]
        leading = tf.range(tf.rank(jac) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(jac)
        new_order = tf.concat([leading, trailing], axis=0)
        jac_T = tf.transpose(jac, new_order)
        # compute metric:
        metric = tf.linalg.matmul(jac_T, jac)
        #
        return metric

    @tf.function()
    def inverse_metric(self, coord):
        """
        Computes the inverse metric at a given point or array of points in (original) parameter space
        """
        # compute Jacobian:
        jac = self.direct_jacobian(coord)
        # take the transpose (we need to calculate the indexes that we want to swap):
        trailing_axes = [-1, -2]
        leading = tf.range(tf.rank(jac) - len(trailing_axes))
        trailing = trailing_axes + tf.rank(jac)
        new_order = tf.concat([leading, trailing], axis=0)
        jac_T = tf.transpose(jac, new_order)
        # compute metric:
        metric = tf.linalg.matmul(jac, jac_T)
        #
        return metric

    @tf.function()
    def coord_metric_derivative(self, coord):
        """
        Compute the coordinate derivative of the metric at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.metric(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def coord_inverse_metric_derivative(self, coord):
        """
        Compute the coordinate derivative of the inverse metric at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.inverse_metric(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def coord_metric_derivative_2(self, coord):
        """
        Compute the second coordinate derivative of the metric at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.coord_metric_derivative(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def coord_inverse_metric_derivative_2(self, coord):
        """
        Compute the second coordinate derivative of the inverse metric at a given point in (original) parameter space
        """
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
            tape.watch(coord)
            f = self.coord_inverse_metric_derivative(coord)
        return tape.batch_jacobian(f, coord)

    @tf.function()
    def levi_civita_connection(self, coord):
        """
        Compute the Levi-Civita connection
        """
        inv_metric = self.inverse_metric(coord)
        metric_derivative = self.coord_metric_derivative(coord)
        # rearrange indexes:
        term_1 = tf.einsum("...kjl -> ...jkl", metric_derivative)
        term_2 = tf.einsum("...lik -> ...ikl", metric_derivative)
        term_3 = tf.einsum("...kli -> ...ikl", metric_derivative)
        # compute
        connection = 0.5*tf.einsum("...ij,...jkl-> ...ikl", inv_metric, term_1 + term_2 - term_3)
        #
        return connection

    @tf.function()
    def geodesic_ode(self, t, y):
        # unpack position and velocity:
        pos = y[:self.num_params]
        vel = y[self.num_params:]
        # compute geodesic equation:
        acc = -tf.einsum("...ijk, ...j, ...k -> ...i", self.levi_civita_connection(tf.convert_to_tensor([pos])), tf.convert_to_tensor([vel]), tf.convert_to_tensor([vel]))
        #
        return tf.concat([vel, acc[0]], axis=0)

    @tf.function()
    def solve_geodesic(self, y_init, yprime_init, solution_times, **kwargs):
        # prepare initial conditions:
        y0 = tf.concat([y_init, yprime_init], axis=0)
        # solve with explicit solver:
        results = tfp.math.ode.DormandPrince(rtol=1.e-4).solve(self.geodesic_ode, initial_time=0., initial_state=y0, solution_times=solution_times, **kwargs)
        #
        return results

    @tf.function()
    def geodesic_distance(self, coord_1, coord_2):
        # map to abstract coordinates:
        abs_coord_1 = self.map_to_abstract_coord(coord_1)
        abs_coord_2 = self.map_to_abstract_coord(coord_2)
        # metric there is Euclidean:
        return tf.linalg.norm(abs_coord_1-abs_coord_2)

    @tf.function()
    def eigenvalue_ode(self, t, y, n):
        # compute metric:
        metric = self.metric(tf.convert_to_tensor([y]))
        # compute eigenvalues:
        _, eigv = tf.linalg.eigh(metric[0])
        #
        return eigv[:, n]

    @tf.function()
    def solve_eigenvalue_ode(self, y0, solution_times, n, **kwargs):
        # solve with explicit solver:
        results = tfp.math.ode.DormandPrince(rtol=1.e-4).solve(self.eigenvalue_ode, initial_time=0., initial_state=y0, solution_times=solution_times, constants={'n': n})
        #
        return results

    ###############################################################################
    # Training statistics:

    def _compute_shift_proba(self):
        zero = np.array(self.Z2X_bijector.inverse(np.zeros(self.num_params, dtype=np.float32)))
        chi2Z0 = np.sum(zero**2)
        pval = scipy.stats.chi2.cdf(chi2Z0, df=self.num_params)
        nsigma = utils.from_confidence_to_sigma(pval)
        return zero, chi2Z0, pval, nsigma

    def _plot_loss(self, ax, logs={}):
        self.log["loss"].append(logs.get('loss'))
        self.log["val_loss"].append(logs.get('val_loss'))
        if ax is not None:
            ax.plot(self.log["loss"], label='Training')
            ax.plot(self.log["val_loss"], label='Testing')
            ax.set_title("Training Loss")
            ax.set_xlabel("Epoch #")
            ax.set_ylabel("Loss")
            ax.legend()

    def _plot_chi2_dist(self, ax, logs={}):
        # Compute chi2 and make sure some are finite
        chi2Z = np.sum(np.array(self.Z2Y_bijector.inverse(self.Y_test))**2, axis=1)
        _s = np.isfinite(chi2Z)
        assert np.any(_s)
        chi2Z = chi2Z[_s]

        # Run KS test
        try:
            # Note that scipy.stats.kstest does not handle weights yet so we need to resample.
            if self.has_weights:
                chi2Z = np.random.choice(chi2Z, size=len(chi2Z), replace=True, p=self.weights_test[_s]/np.sum(self.weights_test[_s]))
            chi2Z_ks, chi2Z_ks_p = scipy.stats.kstest(chi2Z, 'chi2', args=(self.num_params,))
        except:
            chi2Z_ks, chi2Z_ks_p = 0., 0.

        self.log["chi2Z_ks"].append(chi2Z_ks)
        self.log["chi2Z_ks_p"].append(chi2Z_ks_p)

        xx = np.linspace(0, self.num_params*4, 1000)
        bins = np.linspace(0, self.num_params*4, 100)

        # Plot
        if ax is not None:
            ax.plot(xx, scipy.stats.chi2.pdf(xx, df=self.num_params), label='$\\chi^2_{{{}}}$ PDF'.format(self.num_params), c='k', lw=1)
            ax.hist(self.chi2Y, bins=bins, density=True, histtype='step', weights=self.weights_test, label='Pre-gauss ($D_n$={:.3f})'.format(self.chi2Y_ks))
            ax.hist(chi2Z, bins=bins, density=True, histtype='step', weights=self.weights_test[_s], label='Post-gauss ($D_n$={:.3f})'.format(chi2Z_ks))
            ax.set_title(r'$\chi^2_{{{}}}$ PDF'.format(self.num_params))
            ax.set_xlabel(r'$\chi^2$')
            ax.legend(fontsize=8)

    def _plot_chi2_ks_p(self, ax, logs={}):
        # Plot
        if ax is not None:
            ln1 = ax.plot(self.log["chi2Z_ks_p"], label='$p$')
            ax.set_title(r"KS test ($\chi^2$)")
            ax.set_xlabel("Epoch #")
            ax.set_ylabel(r"$p$-value")

            ax2 = ax.twinx()
            ln2 = ax2.plot(self.log["chi2Z_ks"], ls='--', label='$D_n$')
            ax2.set_ylabel('r$D_n$')

            lns = ln1+ln2
            labs = [l.get_label() for l in lns]
            ax2.legend(lns, labs, loc=1)

    def on_epoch_end(self, epoch, logs={}):
        """
        This method is used by Keras to show progress during training if `feedback` is True.
        """
        if self.feedback:
            if isinstance(self.feedback, int):
                if epoch % self.feedback:
                    return
            clear_output(wait=True)
            fig, axes = plt.subplots(1, 3, figsize=(16, 3))
        else:
            axes = [None]*3
        self._plot_loss(axes[0], logs=logs)
        self._plot_chi2_dist(axes[1], logs=logs)
        self._plot_chi2_ks_p(axes[2], logs=logs)

        for k in self.log.keys():
            logs[k] = self.log[k][-1]

        if self.feedback:
            plt.tight_layout()
            plt.show()
            return fig
