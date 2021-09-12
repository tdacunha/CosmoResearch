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

###############################################################################
# run example:

import example_3_generate as example

# run posterior:
analyze_2d_example.run_example_2d(chain=example.posterior_chain,
                                  flow=example.posterior_flow,
                                  param_names=example.posterior_chain.getParamNames().list(),
                                  outroot=example.out_folder+'posterior_',
                                  param_ranges=[[-1.999, 1.999], [-1.999, 1.999]])
