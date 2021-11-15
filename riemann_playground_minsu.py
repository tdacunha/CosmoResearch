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

###############################################################################
# run example:

import example_2_generate as example

def coord_derivative_levi_civita_connection(self, coord):
    """
    """
    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        tape.watch(coord)
        f = self.levi_civita_connection(coord)
    return tape.batch_jacobian(f, coord)


self = example.posterior_flow
coord = self.cast([example.posterior_flow.sample_MAP])

metric = self.metric(coord)
inv_metric = self.inverse_metric(coord)
d_metric = self.coord_metric_derivative(coord)
dinv_metric = self.coord_inverse_metric_derivative(coord)
dd_metric = self.coord_metric_derivative_2(coord)

# assume that d_metric gives (metric index, derivative index)
# if it is (derivative index, metric index), uncomment the next lines
# d_metric = tf.einsum("...ijk -> ...kij", d_metric)
# dinv_metric = tf.einsum("...ijk -> ...kij", dinv_metric)
# dd_metric = tf.einsum("...ijkl -> ...klij", dd_metric)


#dconnection_auto = coord_derivative_levi_civita_connection(self, coord)
# metric has precision loss, symmetrize:
dd_metric = 0.5*(dd_metric + tf.einsum("...mnij->...mnji", dd_metric))

# assume that d_metric gives (metric index, derivative index)
# if it is (derivative index, metric index), uncomment the next lines
# d_metric = tf.einsum("...ijk -> ...kij", d_metric)
# dinv_metric = tf.einsum("...ijk -> ...kij", dinv_metric)

# prepare indexes of first derivatives of the metric:
term_1 = d_metric
term_2 = tf.einsum("...ijk -> ...ikj", d_metric)
term_3 = tf.einsum("...ijk -> ...kij", d_metric)

# compute connection:
connection = 0.5*tf.einsum("...ij,...jkl-> ...ikl", inv_metric, term_1 + term_2 - term_3)
# compute first part of connection derivative:
dconnection = 0.5*tf.einsum("...ijm,...jkl-> ...iklm", dinv_metric, term_1 + term_2 - term_3)
# prepare indexes of second derivatives of the metric:
term_1 = dd_metric
term_2 = tf.einsum("...ijkm -> ...ikjm", dd_metric)
term_3 = tf.einsum("...ijkm -> ...kijm", dd_metric)
# add second term of connection derivative:
dconnection += 0.5*tf.einsum("...ij,...jklm-> ...iklm", inv_metric, term_1 + term_2 - term_3)
# compute connection squared term:
connection2 = tf.einsum("...ijk,...klm-> ...ijlm", connection, connection)
# adjust indexes of connection derivatives:
dconnection = tf.einsum("...ijkl->...lijk", dconnection)
# assemble Riemann tensor:
riemann = tf.einsum("...ijkl-> ...jlik", dconnection) - tf.einsum("...ijkl-> ...jlki", dconnection) \
          + tf.einsum("...ijkl-> ...iljk", connection2) - tf.einsum("...ijkl-> ...ilkj", connection2)
# lower indexes Riemann tensor:
low_riemann = tf.einsum("...mi,...ijkl->...mjkl", metric, riemann)

pippo = connection2 - tf.einsum("...ijkl-> ...ikjl", connection2)


pippo + tf.einsum("...ijkl->...ikjl", pippo)



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
