# -*- coding: utf-8 -*-

###############################################################################
# initial imports:

import os
import numpy as np
import DES_generate
import getdist
from getdist import plots
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec
import color_utilities
import utilities as utils

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
# import the tensiometer tools that we need:
from tensiometer import utilities, gaussian_tension

# import example:
import example_CMB_lensing as example

###############################################################################
# initial settings:

# settings:
contribution_threshold = 0.15
mean_subtract = False
do_exp = True
use_global = False

# output folder:
out_folder = './results/paper_plots/'
if not os.path.exists(out_folder):
    os.mkdir(out_folder)

# latex rendering:
plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
plt.rc('text', usetex=True)

# image size:
x_size = 8.54
y_size = 11.0
main_fontsize = 10.0

# color palette:
colors = [color_utilities.nice_colors(i) for i in range(6)]

###############################################################################
# define chains and flows:

combinations = [
                [example.lcdm_CMB_lensing_log_params_posterior_flow, example.lcdm_CMB_lensing_log_params_prior_flow, example.posterior_chain_lcdm_CMB_lensing, example.prior_chain_lcdm_CMB_lensing],
                ]

num_chains = len(combinations)

###############################################################################
# loop over chains:

for ind in range(num_chains):
    # get chains:
    posterior_flow = combinations[ind][0]
    prior_flow = combinations[ind][1]
    posterior_chain = combinations[ind][2]
    prior_chain = combinations[ind][3]
    # get param names:
    param_names = posterior_flow.param_names
    num_params = len(param_names)
    # print feedback:
    print()
    print('******************************************************************')
    print('*** Doing: ', posterior_flow.name_tag, 'with prior', prior_flow.name_tag)
    print('    with params: ', param_names)
    # get reference point:
    reference_point = posterior_chain.getMeans(pars=[posterior_chain.index[name] for name in param_names])
    # get local fisher:
    fisher = posterior_flow.metric(posterior_flow.cast([reference_point]))[0]
    prior_fisher = prior_flow.metric(prior_flow.cast([reference_point]))[0]
    # compute Neff spectrally:
    C_p = posterior_chain.cov(pars=param_names)
    C_Pi = prior_chain.cov(pars=param_names)
    _temp = np.dot(np.linalg.inv(C_Pi), C_p)
    _eigv, _eigvec = np.linalg.eig(_temp)
    _eigv[_eigv > 1.] = 1.
    _eigv[_eigv < 0.] = 0.
    _Neff = num_params - np.sum(_eigv)
    if use_global:
        fisher = np.linalg.inv(C_p)
        prior_fisher = np.linalg.inv(C_Pi)
    with np.printoptions(precision=2, suppress=True):
        print('    Neff = ', np.sort(1.-_eigv)[::-1], '=', np.round(_Neff, 2))
    # Neff from KL:
    _eig, _eigv = utilities.KL_decomposition(np.linalg.inv(C_p), np.linalg.inv(C_Pi))
    with np.printoptions(precision=2, suppress=True):
        print('    KL eig = ', _eig)
        _eig[_eig < 1.] = 1.
        print('    Neff (KL) = ', np.sort(1.-1./_eig)[::-1], '=', np.round(np.sum(1.-1./_eig), 2))
    # set num modes:
    num_modes = int(np.ceil(_Neff))
    # do local KL decomposition:
    eig, eigv = utilities.KL_decomposition(fisher, prior_fisher)
    sqrt_fisher = scipy.linalg.sqrtm(fisher)
    # sort modes:
    idx = np.argsort(eig)[::-1]
    eig = eig[idx]
    eigv = eigv[:, idx]
    # print out modes:
    temp = np.dot(sqrt_fisher, eigv)
    contributions = temp * temp / eig
    # temp = np.dot(scipy.linalg.sqrtm(np.linalg.inv(fisher)), np.linalg.inv(eigv).T)
    # contributions = eig * temp * temp
    for i in range(num_modes):
        idx_max = np.argmax(contributions[:, i])
        print('    * Mode', i+1)
        print('      Sqrt eig = ', np.round(np.sqrt(eig[i]-1), 2), ' Neff =', np.round(1. - 1./eig[i], 2))
        _directions = np.linalg.inv(eigv).T
        _norm_eigv = _directions[:, i] / _directions[idx_max, i]
        # normalize to fixed param:
        ref_idx = param_names.index('log_sigma8')
        _norm_eigv = _directions[:, i] / _directions[ref_idx, i]
        with np.printoptions(precision=2, suppress=True):
            print('      Variance contributions', contributions[:, i])
        # print out mode:
        string = ''
        for j in range(num_params):
            _name = param_names[j]
            _mean = reference_point[j]
            _mean = '{0:+}'.format(np.round(-_mean, 2))
            _temp = '{0:+.2f}'.format(np.round(_norm_eigv[j], 2))
            string += _temp+'*('+_name+' '+_mean+') '
        print('      Mode =', string, '= 0 +-', '%.2g' % np.sqrt(1./eig[i]/_directions[ref_idx, i]**2))
        # print out filtered mode:
        _contribution_filter = contributions[:, i] > contribution_threshold
        print('      Parameter filter =', _contribution_filter)
        string = ''
        for j in range(num_params):
            if not _contribution_filter[j]:
                continue
            _name = param_names[j]
            _mean = reference_point[j]
            _temp_log = 'log' in _name
            if _temp_log:
                _mean = '-log {0:}'.format(np.round(np.exp(_mean), 2))
            else:
                _mean = '{0:+}'.format(np.round(-_mean, 2))
            _temp = '{0:+.2f}'.format(np.round(_norm_eigv[j], 2))
            string += _temp+'*('+_name+' '+_mean+') '
        _temp_norm_eigv = _norm_eigv
        _temp_norm_eigv[np.logical_not(_contribution_filter)] = 0.0
        _proj_var = np.dot(np.dot(_temp_norm_eigv, np.linalg.inv(fisher)), _temp_norm_eigv)
        print('      Filtered Mode =', string, '= 0 +-', '%.2g' % np.sqrt(_proj_var))
        # compute constraints on filtered mode:
        _filt_names = [name for name, filt in zip(param_names, _contribution_filter) if filt]
        _filt_coeff = _norm_eigv[_contribution_filter]
        _temp_samps = posterior_chain.samples[:, [posterior_chain.index[name] for name in _filt_names]]
        if mean_subtract:
            _temp_samps = _temp_samps - reference_point[_contribution_filter]
        _p_mode = np.dot(_temp_samps, _filt_coeff)
        _log_param = np.any(['log' in name for name in _filt_names])
        _log_param = _log_param and do_exp
        if _log_param:
            _p_mode = np.exp(_p_mode)
        temp_posterior_chain = posterior_chain.copy()
        temp_posterior_chain.addDerived(_p_mode, name='p'+str(i+1), label='p_'+str(i+1))
        temp_posterior_chain.updateBaseStatistics()
        _temp_res = temp_posterior_chain.getLatex('p'+str(i+1))
        if _log_param:
            _temp_res = 'exp '+_temp_res
        print('      ', _temp_res, '+-', '%.2g' % temp_posterior_chain.std(temp_posterior_chain.index['p'+str(i+1)]))
        print(' ')





pass
