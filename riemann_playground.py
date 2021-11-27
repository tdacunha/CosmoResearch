# -*- coding: utf-8 -*-

"""
Run analysis on all examples
"""

###############################################################################
# initial imports:
import os
import numpy as np
import copy
import pickle
import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import tensiometer.gaussian_tension as gaussian_tension
from scipy import optimize

import analyze_2d_example
import KL_analyze_2d_example
import tensorflow as tf
import tensorflow_probability as tfp
import synthetic_probability

###############################################################################
# run example:

import example_1_generate as example


def coord_derivative_levi_civita_connection(self, coord):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(coord)
        f = self.levi_civita_connection(coord)
    return tape.batch_jacobian(f, coord)


def coord_derivative_det_metric(self, coord):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(coord)
        f = self.log_det_metric(coord)
    return tape.gradient(f, coord)


def second_coord_derivative_det_metric(self, coord):
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(coord)
        f = coord_derivative_det_metric(self, coord)
    return tape.batch_jacobian(f, coord)


bij = synthetic_probability.prior_bijector_helper(loc=example.posterior_mean, cov=4.*example.posterior_cov)
flow = synthetic_probability.DiffFlowCallback(example.posterior_chain, prior_bijector=bij, apply_pregauss=False, trainable_bijector=None, param_names=example.posterior_chain.getParamNames().list(), feedback=1)
self = synthetic_probability.TransformedDiffFlowCallback(flow, [tfp.bijectors.Exp(), tfp.bijectors.Exp()])
coord = self.cast([self.sample_MAP+0.1])

P1 = np.linspace(*flow.parameter_ranges['theta_1'], 10)
P2 = np.linspace(*flow.parameter_ranges['theta_2'], 10)
x, y = P1, P2
X, Y = np.meshgrid(x, y)
coord = self.cast(np.exp(np.array([X, Y]).reshape(2, -1).T))

# metric and inverse metric:
metric = self.metric(coord)
inv_metric = self.inverse_metric(coord)
# test metric:
assert np.allclose(np.linalg.inv(metric), inv_metric)  # metric and its inverse
assert np.allclose(tf.einsum("...ij->...ji", metric), metric)  # metric is symmetric
assert np.allclose(tf.einsum("...ij->...ji", inv_metric), inv_metric)  # inverse metric is symmetric
# derivative of the metric:
d_metric = self.coord_metric_derivative(coord)
d_metric = tf.einsum("...ijk->...kij", d_metric)
dinv_metric = self.coord_inverse_metric_derivative(coord)
dinv_metric = tf.einsum("...ijk->...kij", dinv_metric)

# test metric derivatives:
assert np.allclose(d_metric, tf.einsum("...ijk -> ...ikj", d_metric))  # metric indexes are symmetric
assert np.allclose(dinv_metric, tf.einsum("...ijk -> ...ikj", dinv_metric))  # inverse metric indexes are symmetric
assert np.allclose(tf.einsum("...ns, ...amn -> ...ams", inv_metric, d_metric), -tf.einsum("...mn, ...ans -> ...ams", metric, dinv_metric))  # derivative of identity matrix
# second derivative of the metric:
dd_metric = self.coord_metric_derivative_2(coord)
dd_metric = tf.einsum("...ijmn->...mnij", dd_metric)
dd_inv_metric = self.coord_inverse_metric_derivative_2(coord)
dd_inv_metric = tf.einsum("...ijmn->...mnij", dd_metric)
# test metric second derivatives:
assert np.allclose(dd_metric, tf.einsum("...ijkl -> ...ijlk", dd_metric))  # metric indexes are symmetric
assert np.allclose(dd_metric, tf.einsum("...ijkl -> ...jikl", dd_metric))  # derivative indexes are symmetric
# compute connection:
term_1 = tf.einsum("...ij,...kjl-> ...ikl", inv_metric, d_metric)
term_2 = tf.einsum("...ij,...kjl-> ...ilk", inv_metric, d_metric)
term_3 = tf.einsum("...ij,...jkl-> ...ikl", inv_metric, d_metric)
connection = 0.5*(term_1 + term_2 - term_3)
low_connection = tf.einsum("...ij, ...jlm -> ...ilm", metric, connection)
# test connection:
assert np.allclose(connection, tf.einsum("...ijk -> ...ikj", connection))  # last two indexes are symmetric
assert np.allclose(low_connection, tf.einsum("...ijk -> ...ikj", low_connection))  # last two indexes are symmetric
assert np.allclose(d_metric, tf.einsum("...lk, ...lij -> ...ijk", metric, connection) + tf.einsum("...jl, ...lik -> ...ijk", metric, connection))  # metric compatibility of levi civita
assert np.allclose(d_metric, tf.einsum("...ikj->...jik", low_connection) + tf.einsum("...kij->...jik", low_connection))  # metric compatibility
assert np.allclose(connection, self.levi_civita_connection(coord))  # compatibility with built in method
assert np.allclose(tf.einsum("...iki", connection), 0.5 * coord_derivative_det_metric(self, coord))  # identity in https://en.wikipedia.org/wiki/List_of_formulas_in_Riemannian_geometry
# compute derivative of connection:
term_1 = tf.einsum("...ra,...mnas-> ...mrns", inv_metric, dd_metric)
term_2 = tf.einsum("...ra,...msan-> ...mrns", inv_metric, dd_metric)
term_3 = tf.einsum("...ra,...mans-> ...mrns", inv_metric, dd_metric)
term_4 = tf.einsum("...mra,...nas-> ...mrns", dinv_metric, d_metric)
term_5 = tf.einsum("...mra,...san-> ...mrns", dinv_metric, d_metric)
term_6 = tf.einsum("...mra,...ans-> ...mrns", dinv_metric, d_metric)
dconnection = 0.5*(term_1 + term_2 - term_3 + term_4 + term_5 - term_6)

dconnection = coord_derivative_levi_civita_connection(self, coord)





# compute connection squared:
connection2 = tf.einsum("...ijk,...klm-> ...ijlm", connection, connection)
# compute riemann:
riemann = tf.einsum("...mrns->...rsmn", dconnection) - tf.einsum("...nrms->...rsmn", dconnection) + tf.einsum("...rmns->...rsmn", connection2) - tf.einsum("...rnms->...rsmn", connection2)
low_riemann = tf.einsum("...mi,...ijkl->...mjkl", metric, riemann)
low_riemann
riemann


ricci = tf.einsum("...rsrn", riemann)

ricci_scalar = tf.einsum("...ij,...ij->...", inv_metric, ricci)
ricci_scalar


ricci

low_riemann + tf.einsum("...abcd->...abdc", low_riemann)

low_riemann + tf.einsum("...abcd->...bacd", low_riemann)

low_riemann + tf.einsum("...abcd->...acdb", low_riemann) + tf.einsum("...abcd->...adbc", low_riemann)

low_riemann - tf.einsum("...abcd->...cdab", low_riemann)







ricci = tf.einsum("...ijil", riemann)

ricci_scalar = tf.einsum("...ij,...ij->...", ricci, inv_metric)

riemann

ricci

ricci_scalar


pass
