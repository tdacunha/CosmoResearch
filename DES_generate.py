# -*- coding: utf-8 -*-

"""
Read in and cache the flows for DES in LCDM.
"""

###############################################################################
# initial imports:
import os
import numpy as np

import sys
here = './'
temp_path = os.path.realpath(os.path.join(os.getcwd(), here+'tensiometer'))
sys.path.insert(0, temp_path)
import getdist
from getdist import plots, MCSamples
from getdist.gaussian_mixtures import GaussianND
import analyze_2d_example
import importlib
import pickle

import synthetic_probability
from tensorflow.keras.callbacks import ReduceLROnPlateau
callbacks = [ReduceLROnPlateau()]
from tensorflow.keras.initializers import TruncatedNormal, Zeros, GlorotNormal

###############################################################################
# chains common settings:

# load the chains (remove no burn in since the example chains have already been cleaned):
chains_dir = here+'/chains/'
# the DES chain:
settings = {'ignore_rows': 0.3, 'smooth_scale_1D': 0.3, 'smooth_scale_2D': 0.3}

###############################################################################
# helper to load chains and train:


def helper_load_chains(param_names, prior_chain, posterior_chain, flow_cache, **kwargs):

    # initialize:
    num_params = len(param_names)
    nsamples = 100000

    # create cache directory:
    if not os.path.exists(flow_cache):
        os.mkdir(flow_cache)

    # get parameters for flows:
    batch_size = kwargs.get('batch_size', None)
    epochs = kwargs.get('epochs', 100)
    steps_per_epoch = kwargs.get('steps_per_epoch', None)
    n_maf = kwargs.get('n_maf', 1*num_params)
    hidden_units = kwargs.get('hidden_units', [num_params]*2)
    kernel_initializer = kwargs.get('kernel_initializer', GlorotNormal())

    # prior flow:
    if os.path.isfile(flow_cache+'/prior'+'_permutations.pickle'):
        # load trained model:
        temp_MAF = synthetic_probability.SimpleMAF.load(num_params, flow_cache+'/prior', n_maf=n_maf, hidden_units=hidden_units)
        # initialize flow:
        prior_flow = synthetic_probability.DiffFlowCallback(prior_chain, prior_bijector='ranges', trainable_bijector=temp_MAF.bijector,
                                                            param_names=param_names, feedback=0)
    else:
        # initialize prior flow:
        prior_flow = synthetic_probability.DiffFlowCallback(prior_chain, prior_bijector='ranges',
                                                            param_names=param_names, feedback=1, n_maf=n_maf,
                                                            hidden_units=hidden_units)
        # train prior flow:
        prior_flow.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        # save trained model:
        prior_flow.MAF.save(flow_cache+'/prior')
        # plot:
        g = plots.get_subplot_plotter()
        g.triangle_plot([prior_chain, prior_flow.MCSamples(nsamples)], params=param_names, filled=False)
        g.export(flow_cache+'/0_learned_prior_distribution.pdf')

    # posterior flow:
    if os.path.isfile(flow_cache+'/posterior'+'_permutations.pickle'):
        # load trained model:
        temp_MAF = synthetic_probability.SimpleMAF.load(num_params, flow_cache+'/posterior', n_maf=n_maf, hidden_units=hidden_units)
        # initialize flow:

        posterior_flow = synthetic_probability.DiffFlowCallback(posterior_chain,
                                                                #prior_bijector=prior_flow.bijector,
                                                                trainable_bijector=temp_MAF.bijector,
                                                                param_ranges=prior_flow.parameter_ranges,
                                                                param_names=prior_flow.param_names,
                                                                feedback=1,)
    else:
        # initialize posterior flow:
        posterior_flow = synthetic_probability.DiffFlowCallback(posterior_chain,
                                                                #prior_bijector=prior_flow.bijector,
                                                                #prior_bijector=None,
                                                                param_ranges=prior_flow.parameter_ranges,
                                                                param_names=prior_flow.param_names,
                                                                feedback=1,
                                                                n_maf=n_maf, hidden_units=hidden_units, kernel_initializer=kernel_initializer) #kernel_initializer=TruncatedNormal(stddev=1e-3))
        # train posterior flow:
        posterior_flow.global_train(pop_size = 5, batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        #posterior_flow.train(batch_size=batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks)
        # save trained model:
        posterior_flow.MAF.save(flow_cache+'/posterior')
        # plot posterior:
        g = plots.get_subplot_plotter()
        g.triangle_plot([posterior_chain, posterior_flow.MCSamples(nsamples)], params=param_names, filled=False)
        g.export(flow_cache+'/0_learned_posterior_distribution.pdf')

    ## find MAP of flows:
    #if os.path.isfile(flow_cache+'/prior_MAP.pickle'):
    #    temp = pickle.load(open(flow_cache+'/prior_MAP.pickle', 'rb'))
    #    prior_flow.MAP_coord = temp['MAP_coord']
    #    prior_flow.MAP_logP = temp['MAP_logP']
    #else:
    #    # find map:
    #    res = prior_flow.fast_MAP_finder()
    #    # store:
    #    temp = {
    #        'MAP_coord': prior_flow.MAP_coord,
    #        'MAP_logP': prior_flow.MAP_logP,
    #        }
    #    # save out:
    #    pickle.dump(temp, open(flow_cache+'/prior_MAP.pickle', 'wb'))
    #    # plot:
    #    g = plots.get_subplot_plotter()
    #    markers = [[i, j] for i, j in zip(prior_flow.MAP_coord, prior_flow.chain_MAP)]
    #    g.triangle_plot([prior_chain, prior_flow.MCSamples(nsamples)], params=param_names, filled=False, markers=markers)
    #    g.export(flow_cache+'/0_learned_prior_distribution_MAP.pdf')
    ##
    #if os.path.isfile(flow_cache+'/posterior_MAP.pickle'):
    #    temp = pickle.load(open(flow_cache+'/posterior_MAP.pickle', 'rb'))
    #    posterior_flow.MAP_coord = temp['MAP_coord']
    #    posterior_flow.MAP_logP = temp['MAP_logP']
    #else:
    #    # res = posterior_flow.MAP_finder(maxiter = 200, disp=True, x0 = posterior_flow.sample_MAP)
    #    # res = posterior_flow.MAP_finder(maxiter = 200, disp=True, x0 = means)
    #    res = posterior_flow.fast_MAP_finder()
    #    # store:
    #    temp = {
    #        'MAP_coord': posterior_flow.MAP_coord,
    #        'MAP_logP': posterior_flow.MAP_logP,
    #        }
    #    # save out:
    #    pickle.dump(temp, open(flow_cache+'/posterior_MAP.pickle', 'wb'))
    #    # plot:
    #    g = plots.get_subplot_plotter()
    #    markers = [[i, j] for i, j in zip(posterior_flow.MAP_coord, posterior_flow.chain_MAP)]
    #    markers = [i for i in posterior_flow.MAP_coord]
    #    g.triangle_plot([posterior_chain, posterior_flow.MCSamples(nsamples)], params=param_names, filled=False, markers=markers)
    #    g.export(flow_cache+'/0_learned_posterior_distribution_MAP.pdf')

    return prior_flow, posterior_flow


if __name__ == '__main__':

    # test example:

    # output folder:
    out_folder = './results/example_des/'
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)

    # load chains:
    prior_chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'001_DESY1_3x2_prior', no_cache=True, settings=settings)
    posterior_chain = getdist.mcsamples.loadMCSamples(file_root=chains_dir+'001_DESY1_3x2', no_cache=True, settings=settings)

    param_names = ['omegam', 'omegab', 'H0', 'sigma8', 'ns',
                   'DES_b1', 'DES_b2', 'DES_b3', 'DES_b4', 'DES_b5',
                   'DES_m1', 'DES_m2', 'DES_m3', 'DES_m4',
                   'DES_AIA', 'DES_alphaIA',
                   'DES_DzL1', 'DES_DzL2', 'DES_DzL3', 'DES_DzL4', 'DES_DzL5',
                   'DES_DzS1', 'DES_DzS2', 'DES_DzS3', 'DES_DzS4',]
    flow_cache = out_folder + '/001_DESY1_3x2_cache/'
    kwargs = {}
    prior_flow, posterior_flow = helper_load_chains(param_names, prior_chain, posterior_chain, flow_cache)







pass
