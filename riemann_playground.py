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
#dconnection_auto = coord_derivative_levi_civita_connection(self, coord)
# metric has precision loss, symmetrize:
dd_metric = 0.5*(dd_metric + tf.einsum("...ij->...ji", dd_metric))

# prepare indexes of first derivatives of the metric:
term_1 = tf.einsum("...kil -> ...ikl", d_metric)
term_2 = tf.einsum("...lik -> ...ikl", d_metric)
term_3 = tf.einsum("...kli -> ...ikl", d_metric)
# compute connection:
connection = 0.5*tf.einsum("...ij,...jkl-> ...ikl", inv_metric, term_1 + term_2 - term_3)
# compute first part of connection derivative:
dinv_metric = tf.einsum("...ijk->...kij", dinv_metric)
dconnection = 0.5*tf.einsum("...mij,...jkl-> ...mikl", dinv_metric, term_1 + term_2 - term_3)
# prepare indexes of second derivatives of the metric:
term_1 = tf.einsum("...kilm -> ...mikl", dd_metric)
term_2 = tf.einsum("...likm -> ...mikl", dd_metric)
term_3 = tf.einsum("...klim -> ...mikl", dd_metric)
# add second term of connection derivative:
dconnection += 0.5*tf.einsum("...ij,...jkl-> ...ikl", inv_metric, term_1 + term_2 - term_3)
# compute connection squared term:
connection2 = tf.einsum("...ijk,...klm-> ...ijlm", connection, connection)
# adjust indexes of connection derivatives:
dconnection = tf.einsum("...ijkl->...kilj", dconnection)
# assemble Riemann tensor:
riemann = dconnection - tf.einsum("...ijkl-> ...ikjl", dconnection) \
          + connection2 - tf.einsum("...ijkl-> ...ikjl", connection2)
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
