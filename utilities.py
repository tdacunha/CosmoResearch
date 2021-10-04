"""
Things that we will add to tensiometer utilities
"""

import numpy as np
from getdist import plots

def covariance_around(samples, center, weights=None):
    """
    Compute second moment around point
    """
    # number of samples and number of parameters:
    nsamps, npar = samples.shape
    # shift samples:
    diffs = samples - center
    diffs = diffs.T
    # initialize weights:
    if weights is None:
        weights = np.ones(nsamps)
    # do the calculation:
    cov = np.empty((npar, npar))
    for i, diff in enumerate(diffs):
        weightdiff = diff * weights
        for j in range(i, npar):
            cov[i, j] = weightdiff.dot(diffs[j])
            cov[j, i] = cov[i, j]
    cov /= np.sum(weights)
    #
    return cov


def simple_triangle_plot(triangle_names, all_chains, all_colors, all_labels, g=None):
    # create plot:
    if g is None:
        g = plots.getSubplotPlotter();
    plot_col = len(triangle_names)
    g.make_figure(nx=plot_col, ny=plot_col-1, sharex=g.settings.no_triangle_axis_labels,
                  sharey=g.settings.no_triangle_axis_labels)
    # diagonal part of the plot:
    bottom = len(triangle_names) - 1
    #lims = []
    for i, param in enumerate(triangle_names):
        for i2 in range(bottom, i, -1):
            g._subplot(i, i2, pars=(triangle_names, triangle_names[i2]),
                       sharex=g.subplots[bottom, i] if i2 != bottom else None,
                       sharey=g.subplots[i2, 0] if i > 0 else None)
    #    ax = g._subplot(i, i, pars=(param,), sharex=g.subplots[bottom, i] if i != bottom else None)
    #    g._inner_ticks(ax, False)
    #    chains = [ch for ch in all_chains if param in ch.getParamNames().list()]
    #    colors = [all_colors[ind] for ind, ch in enumerate(all_chains) if param in ch.getParamNames().list()]
    #    xlim = g.plot_1d(chains, param, do_xlabel=i == plot_col - 1,
    #                     no_label_no_numbers=g.settings.no_triangle_axis_labels,
    #                     label_right=True, no_zero=True, no_ylabel=True, no_ytick=True,
    #                     ax=ax, _ret_range=True, colors=colors)
    #    # add marker for best fit:
    #    #for ch, col in zip(chains, colors):
    #    #    try:
    #    #        ax.axvline(ch.getBestFit().parWithName(param).best_fit, color=col, ls='--', lw=1.)
    #    #    except:
    #    #        pass
    #    lims.append(xlim)
    # non diagonal part of the plot:
    for i, param in enumerate(triangle_names):
        for i2 in range(i + 1, len(triangle_names)):
            param2 = triangle_names[i2]
            pair = [param, param2]
            ax = g.subplots[i2, i]
            chains = [ch for ch in all_chains if param in ch.getParamNames().list() and param2 in ch.getParamNames().list()]
            colors = [all_colors[ind] for ind, ch in enumerate(all_chains) if param in ch.getParamNames().list() and param2 in ch.getParamNames().list()]
            if len(chains) == 0:
                continue
            g.plot_2d(chains[::-1], param_pair=pair, do_xlabel=i2 == plot_col - 1, do_ylabel=i == 0,
                      no_label_no_numbers=g.settings.no_triangle_axis_labels, shaded=False,
                      add_legend_proxy=i == 0 and i2 == 1, ax=ax, colors=colors[::-1], filled=True)
            g._inner_ticks(ax)
            # add marker for best fit:
            #for ch, col in zip(chains, colors):
            #    try:
            #        ax.axvline(ch.getBestFit().parWithName(param).best_fit, color=col, ls='--', lw=1.)
            #        ax.axhline(ch.getBestFit().parWithName(param2).best_fit, color=col, ls='--', lw=1.)
            #        ax.scatter([ch.getBestFit().parWithName(param).best_fit], [ch.getBestFit().parWithName(param2).best_fit], c=col, edgecolors='white')
            #    except:
            #        pass
            # limits:
            #if i != i2:
            #    ax.set_ylim(lims[i2])
            #if i2 == bottom:
            #    ax.set_xlim(lims[i])

    g.finish_plot(all_labels, label_order=None,
                  legend_ncol=None or g.settings.figure_legend_ncol, legend_loc=None,
                  no_extra_legend_space=False)
    #
    return g











pass
