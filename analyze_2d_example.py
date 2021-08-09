# -*- coding: utf-8 -*-

"""
General code to analyze examples

For testing purposes:

import example_2_generate as example

posterior_chain = example.posterior_chain
prior_chain = example.prior_chain
param_names = posterior_chain.getParamNames().list()
outroot = example.out_folder
"""

###############################################################################
# initial imports:
import os
import numpy as np

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import synthetic_probability
from tensorflow.keras.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau()]
from getdist import plots, MCSamples

###############################################################################

def run_example_2d(posterior_chain, prior_chain, param_names, outroot):
    """
    Run full analysis of 2d example case, as in flow playground
    """

    # obtain the syntetic probability:
    flow_P = synthetic_probability.DiffFlowCallback(posterior_chain, param_names=param_names, feedback=1, learning_rate=0.01)
    batch_size = 8192
    epochs = 40
    steps_per_epoch = 128
    flow_P.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)

    # plot learned distribution from samples:
    N = 10000
    X_sample = np.array(flow_P.sample(N))
    flow_chain = MCSamples(samples=X_sample, names=param_names, label='Learned distribution')

    g = plots.get_subplot_plotter()
    g.triangle_plot([posterior_chain, flow_chain], params=param_names, filled=False)
    g.export(outroot+'1_learned_posterior_distribution.pdf')


    pass





pass
