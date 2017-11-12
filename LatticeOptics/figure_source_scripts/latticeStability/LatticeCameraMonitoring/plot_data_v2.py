import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pylab as pl
from plot_tools import *
import matplotlib as mpl
import brewer2mpl as cb
from matplotlib import rcParams
import brewer2mpl as cb
import cmocean as cm
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines


def fit_hist_gauss(data, **kwargs):
    guess = kwargs.pop('guess', [0, 1])
    kwargs.update({'density': True})
    fit_fun = lambda x, mu, sig: 1.0/np.sqrt(2*np.pi*np.square(sig))*np.exp(-1*np.divide(np.square(x-mu),
                                                                                        2*np.square(sig)))
    weights, bin_edges = np.histogram(data, **kwargs)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    popt, pcov = curve_fit(fit_fun, bin_centers, weights, p0=guess)
    return popt, np.sqrt(np.diag(pcov)), fit_fun


if __name__ == "__main__":
    calibration_data = pd.read_csv('calibration_data_comparison.csv')
    oos_data = pd.read_csv('out_of_sample_data_with_p20_s4.csv')
    xdiffs = oos_data.diff()[1:][['x_t', 'x_l']]
    n_step=5
    xdiffs5 = oos_data.diff(n_step)[1:][['x_t', 'x_l']][n_step::n_step]
    print xdiffs.corr()
    xtfit = fit_hist_gauss(xdiffs.x_t)
    xlfit = fit_hist_gauss(xdiffs.x_l)
    print xdiffs['x_t'].autocorr(lag=1)
    print xdiffs['x_l'].autocorr(lag=1)
    mean_xt, mean_xl = np.mean(oos_data.x_t), np.mean(oos_data.x_l)
    print xdiffs5.x_t
    print fit_hist_gauss(xdiffs5.x_t)

    fig_name = "pointing_figure"
    rcParams.update({})
    scl = 1/1.61
    fig_width = 5
    fig_height = scl*fig_width


    rcpars.update({
        #"figure.autolayout": True,
        "figure.figsize": (fig_width, fig_height),
        #"figure.subplot.left": .1,
        #"figure.subplot.right": .94,
        #"figure.subplot.bottom": .05,
        #"figure.subplot.top": .95,
    })

    pl.rcParams.update(rcpars)

    labels = ['b', 'c', 'a']
    init_plt()

    gs0_bounds = {'wspace': 0.05}
    gs0 = gridspec.GridSpec(4, 5, wspace=.1, hspace=.1)
    gs0.update(**gs0_bounds)

    fig = pl.figure()
    ax0 = plt.subplot(gs0[2:, 0:2])
    symbols = ['1', '2', '3', '4', '1', '2', '3', '4', '1', '2', '3']# 's', 'p', 'h', '+', 'x', 'D', 'o']
    red_mkr = mlines.Line2D([], [], color=color_or_rd, marker='o', markersize=3, label='atoms', linewidth=0)
    gray_mkr = mlines.Line2D([], [], color=color_grey, marker='o', markersize=3, label='camera', linewidth=0)
    for i, sym in zip(range(len(calibration_data.index)), symbols):
        ax0.plot(calibration_data.x_t[i] - mean_xt, calibration_data.x_l[i] - mean_xl, marker=sym, color=color_or_rd)
        ax0.plot(calibration_data.x_ct[i] - mean_xt, calibration_data.x_cl[i] - mean_xl, marker=sym, color=color_grey)

    ax0.legend(handles=[red_mkr, gray_mkr], numpoints=1, loc=3, fontsize=5, labelspacing=0, columnspacing=0)
    ax0.set_xlim([-5, 5])
    ax0.set_ylim([-6, 4])
    ax0.set_xlabel('$x_t$ (sites)')
    ax0.set_ylabel('$x_l$ (sites)')
    ax0.set_aspect('equal')

    xy = (-.6, 1.02)
    ax0.annotate(labels.pop(0), xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    ax2 = plt.subplot(gs0[0:2, 2:])
    ax3 = plt.subplot(gs0[2:, 2:])

    ax2.plot(oos_data.x_t - mean_xt, color=color_or_rd)
    ax2.plot(oos_data.x_ct - mean_xt, color=color_grey)
    ax2.set_ylabel(r'$x_t$')
    ax2.set_ylim([-3, 3])
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.plot(oos_data.x_l - mean_xl, color=color_or_rd)
    ax3.plot(oos_data.x_cl - mean_xl, color=color_grey)
    ax3.set_ylabel(r'$x_l$')
    ax3.set_ylim([-3, 3])
    ax3.set_xlabel('Experimental realization index')

    xy = (-.2, 1.02)
    ax2.annotate(labels.pop(0), xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    ax4 = plt.subplot(gs0[0:2, 0:2])

    # HISTOGRAM SETTINGS
    settings = {}
    n_bins = settings.pop('n_bins', 21)
    hist_type = settings.pop('hist_type', 'stepfilled')
    is_normed = settings.pop('is_normed', True)
    vlines = settings.pop('vlines', [])
    xlabel = settings.pop('xlabel', None)
    ylabel = settings.pop('ylabel', None)
    xlim = settings.pop('xlim', (-3, 3))
    title = settings.pop('title', None)
    major_tick_dx = settings.pop('major_tick_dx', 5)
    major_tick_dy = settings.pop('major_tick_dy', 5)
    minor_tick_dx = settings.pop('minor_tick_dx', 1)
    minor_tick_dy = settings.pop('minor_tick_dy', 1)
    color = settings.pop('color', 'blue')

    n, bins, _ = ax4.hist(xdiffs.x_t, bins=n_bins, range=xlim, edgecolor=color_black, color=cb.get_map('BuGn', 'Sequential', 9).mpl_colors[-1], histtype=hist_type,
                         normed=is_normed, alpha=.75, label=r"$x_t$")
    ax4.plot(np.linspace(np.min(bins), np.max(bins), 100),
             xtfit[2](np.linspace(np.min(bins), np.max(bins), 100), *list(xtfit[0])),
             linewidth=1, color = cb.get_map('BuGn', 'Sequential', 9).mpl_colors[-1])

    n, bins, _ = ax4.hist(xdiffs.x_l, bins=n_bins, range=xlim, edgecolor=color_black, color=cb.get_map('PuBu', 'Sequential', 9).mpl_colors[-1], histtype=hist_type,
                         normed=is_normed, label=r'$x_l$', alpha=0.75)
    ax4.plot(np.linspace(np.min(bins), np.max(bins), 100),
             xlfit[2](np.linspace(np.min(bins), np.max(bins), 100), *list(xlfit[0])),
             linewidth=1, color=cb.get_map('PuBu', 'Sequential', 9).mpl_colors[-1])

    print xtfit
    print xlfit


    ax4.legend(loc=2)

    ax4.set_ylabel(r'$P(\delta x)$')
    ax4.set_xlabel(r'$\delta x$')

    ax4.set_xlim([-3, 3])
    #plt.setp(ax4.get_xticklabels(), visible=False)
    gs0.tight_layout(fig)

    xy = (-.3, 1.02)
    ax4.annotate(labels.pop(0), xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    #ax.grid(zorder=-1, color=color_grey)
    #ax.grid()
    plt.savefig('{0}.pdf'.format(fig_name))