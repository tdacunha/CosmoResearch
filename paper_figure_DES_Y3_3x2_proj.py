# -*- coding: utf-8 -*-

###############################################################################
# initial imports:

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.integrate import simps
from getdist import plots, MCSamples
import color_utilities
import getdist
getdist.chains.print_load_details = False
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

# add path for correct version of tensiometer:
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
from tensiometer import utilities
import synthetic_probability

###############################################################################
# initial settings:

import example_DES_Y3 as example

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# color palette:
colors = [color_utilities.nice_colors(i) for i in range(6)]

###############################################################################
# general utility functions:


# helper function to get levels:
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
        try:
            res = optimize.brentq(lambda x: _helper(x)-c, np.amin(P), np.amax(P))
            levs.append(res)
        except:
            print('Cannot generate proper levels')
            levs = len(conf)
            break
    levs = np.sort(levs)
    return levs


##############################################################################
# compute projections:

# LCDM:
p = example.posterior_chain_lcdm_3x2.getParams()
example.posterior_chain_lcdm_3x2.addDerived(p.sigma8*p.omegam**1.0, 'p1')

# mnu:
p = example.posterior_chain_mnu_3x2.getParams()
example.posterior_chain_mnu_3x2.addDerived(p.sigma8*p.omegam**1.0, 'p1')

# wCDM:
p = example.posterior_chain_wcdm_3x2.getParams()
example.posterior_chain_wcdm_3x2.addDerived(p.sigma8*p.omegam**1.0, 'p1')

###############################################################################
# plot:

levels = [utilities.from_sigma_to_confidence(i) for i in range(2, 0, -1)]

x_size = 18.0 / 2
y_size = 9.0
main_fontsize = 10.0

# start the plot:
fig = plt.gcf()
fig.set_size_inches(x_size/2.54, y_size/2.54)
gs = gridspec.GridSpec(1, 1)
ax1 = plt.subplot(gs[0])

##############################################################################
# first panel LCDM:
##############################################################################

density = example.posterior_chain_lcdm_3x2.get1DDensity('p1', normalized=True)
density.P = density.P / simps(density.P, density.x)

ax1.plot(density.x, density.P)

density = example.posterior_chain_mnu_3x2.get1DDensity('p1', normalized=True)
density.P = density.P / simps(density.P, density.x)

ax1.plot(density.x, density.P)


density = example.posterior_chain_wcdm_3x2.get1DDensity('p1', normalized=True)
density.P = density.P / simps(density.P, density.x)

ax1.plot(density.x, density.P)





pass
