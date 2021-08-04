"""

TODO:

- save ranges from MCSamples
- implement MAP finder

"""

###############################################################################
# initial imports and set-up:

import os
import time
import gc
from numba import jit
import numpy as np
import getdist.chains as gchains
gchains.print_load_details = False
from getdist import MCSamples, WeightedSamples
import scipy
from scipy.linalg import sqrtm
from scipy.integrate import simps
from scipy.spatial import cKDTree
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
    :param early_stop_nsigma: absolute error on the tension at the zero-shift point to be used
        as an approximate convergence criterion for early stopping, defaults to 0.
    :type early_stop_nsigma: float, optional
    :param early_stop_patience: minimum number of epochs to use when `early_stop_nsigma` is non-zero, defaults to 10.
    :type early_stop_patience: int, optional
    :raises NotImplementedError: if `pregauss_bijector` is not None.
    :reference: George Papamakarios, Theo Pavlakou, Iain Murray (2017). Masked Autoregressive Flow for Density Estimation. `arXiv:1705.07057 <https://arxiv.org/abs/1705.07057>`_
    """

    def __init__(self, chain, param_names=None, Z2Y_bijector='MAF', pregauss_bijector=None, learning_rate=1e-3, feedback=1, validation_split=0.1, early_stop_nsigma=0., early_stop_patience=10, **kwargs):

        self.feedback = feedback

        # Chain
        self._init_chain(chain, param_names=param_names, validation_split=validation_split)

        # Transformed distribution
        self._init_transf_dist(Z2Y_bijector, learning_rate=learning_rate, **kwargs)
        if feedback > 0:
            print("Building flow")
            print("    - trainable parameters:", self.model.count_params())

        # Metrics
        keys = ["loss", "val_loss", "shift0_chi2", "shift0_pval", "shift0_nsigma", "chi2Z_ks", "chi2Z_ks_p"]
        self.log = {_k: [] for _k in keys}

        self.chi2Y = np.sum(self.Y_test**2, axis=1)
        self.chi2Y_ks, self.chi2Y_ks_p = scipy.stats.kstest(self.chi2Y, 'chi2', args=(self.num_params,))

        # Options
        self.early_stop_nsigma = early_stop_nsigma
        self.early_stop_patience = early_stop_patience

        # Pre-gaussianization
        if pregauss_bijector is not None:
            # The idea is to introduce yet another step of deterministic gaussianization, eg using the prior CDF
            # or double prior (convolved with itself, eg a triangular distribution)
            raise NotImplementedError

        # internal variables:
        self.is_trained = False

    def _init_chain(self, chain, param_names=None, validation_split=0.1):
        # initialize param names:
        if param_names is None:
            param_names = chain.getParamNames().getRunningNames()
        else:
            chain_params = chain.getParamNames().list()
            if not np.all([name in chain_params for name in param_names]):
                raise ValueError('Input parameter is not in the diff chain.\n',
                                 'Input parameters ', param_names, '\n'
                                 'Possible parameters', chain_params)
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
        # We're trying to loop through the full sample each epoch
        tf.random.set_seed(1)
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
        self.is_trained = True
        return hist

    def MAP_finder():
        """
        """
        pass

    ###############################################################################
    # Information geometry methods:

    def log_probability():
        """
        """
        pass

    def grad_tape(self, coords_z):
        """
        Make this a class property for the two bijectors
        """
        bijector = self.Z2X_bijector
        delta = tf.Variable([0.0, 0.0])
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as grad:
            grad.watch(delta)
            f = bijector(coords_z+delta)
        self.grad = grad

        return grad, f, delta

    def grad_tape_inv(self, coords_z):
        bijector_inv = self.Z2X_bijector.inverse
        delta_inv = tf.Variable([0.0, 0.0])
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as grad_inv:
            grad_inv.watch(delta_inv)
            f_inv = bijector_inv(coords_z+delta_inv)
        self.grad_inv = grad_inv

        return grad_inv, f_inv, delta_inv

    def jacobian(self, coords_z):
        bijector = self.Z2X_bijector
        grad, f, delta = self.grad_tape(coords_z)
        jac = grad.jacobian(f,delta)
        self.jac = jac
        return jac


    def jacobian_inverse(self, coords):
        """
        """
        bijector = self.Z2X_bijector
        bijector_inv = bijector.inverse

        pass

    def jacobian_T(self, coords_z):
        """
        Remove?
        """
        bijector = self.Z2X_bijector
        jac = self.jacobian(coords_z)
        jac_T = tf.transpose(jac, (0,2,1))
        self.jac_T = jac_T
        return jac_T

    def metric(self, coords):
        bijector = self.Z2X_bijector
        coords_z = np.array((bijector.inverse)(coords.astype(np.float32)))
        jac = self.jacobian(coords_z)
        jac_T = self.jacobian_T(coords_z)
        metric = tf.linalg.matmul(jac, jac_T)
        self.metric = metric
        return metric

    def inverse_metric(coords_x):
        """
        Fill in
        """
        pass

    def det_metric(self, x_array, y_array):
        """
        """
        bijector = self.Z2X_bijector

        log_det = -bijector.forward_log_det_jacobian(coords_z, event_ndims=1)
        log_det = np.array(log_det).T
        det = (np.exp(log_det))**2
        return det

    def hessian(self, coords_z):
        """
        Make this a class method?
        """
        bijector = self.Z2X_bijector
        delta = tf.Variable([0.0,0.0])
        with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t2:
            t2.watch(delta)
            with tf.GradientTape(watch_accessed_variables=False, persistent=True) as t1:
                t1.watch(delta)
                f = bijector(coords_z+delta)
            g = t1.jacobian(f,delta)
        h = t2.jacobian(g,delta)
        self.hessian = h
        return h

    def coord_metric_derivative():
        """
        \partial_\mu g_\\alpha\\beta
        """
        pass


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

    def _plot_shift_proba(self, ax, logs={}):
        # Compute chi2 at zero shift
        zero, chi2Z0, pval, nsigma = self._compute_shift_proba()
        self.log["shift0_chi2"].append(chi2Z0)
        self.log["shift0_pval"].append(pval)
        self.log["shift0_nsigma"].append(nsigma)

        # Plot
        if ax is not None:
            ax.plot(self.log["shift0_chi2"])
            ax.set_title(r"$\chi^2$ at zero-shift")
            ax.set_xlabel("Epoch #")
            ax.set_ylabel(r"$\chi^2$")

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
            fig, axes = plt.subplots(1, 4, figsize=(16, 3))
        else:
            axes = [None]*4
        self._plot_loss(axes[0], logs=logs)
        self._plot_shift_proba(axes[1], logs=logs)
        self._plot_chi2_dist(axes[2], logs=logs)
        self._plot_chi2_ks_p(axes[3], logs=logs)

        for k in self.log.keys():
            logs[k] = self.log[k][-1]

        if self.early_stop_nsigma > 0.:
            if len(self.log["shift0_nsigma"]) > self.early_stop_patience and \
               np.std(self.log["shift0_nsigma"][-self.early_stop_patience:]) < self.early_stop_nsigma and \
               self.log["chi2Z_ks_p"][-1] > 1e-6:
                self.model.stop_training = True

        if self.feedback:
            plt.tight_layout()
            plt.show()
            return fig