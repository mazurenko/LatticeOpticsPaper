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
    n_step = 5
    xdiffs5 = oos_data.diff(n_step)[1:][['x_t', 'x_l']][n_step::n_step]
    print xdiffs.corr()
    xtfit = fit_hist_gauss(xdiffs.x_t)
    xlfit = fit_hist_gauss(xdiffs.x_l)
    print xtfit
    print xlfit
    print xdiffs['x_t'].autocorr(lag=1)
    print xdiffs['x_l'].autocorr(lag=1)
    mean_xt, mean_xl = np.mean(oos_data.x_t), np.mean(oos_data.x_l)
    print xdiffs5.x_t
    #print "5 site correlations"
    #print "transverse step 5 shots: {} +/- {}".format(fit_hist_gauss(xdiffs5.x_t)[0][1], fit_hist_gauss(xdiffs5.x_t)[1][1]),
    #print "lateral step 5 shots: {} +/- {}".format(fit_hist_gauss(xdiffs5.x_l)[0][1], fit_hist_gauss(xdiffs5.x_l)[1][1]),

    n_shots = 1000
    print "transverse step estimated after n shots: {} +/- {}".format(np.sqrt(n_shots/5)*fit_hist_gauss(xdiffs5.x_t)[0][1], np.sqrt(n_shots/5)*fit_hist_gauss(xdiffs5.x_t)[1][1]),
    print "transverse step estimated after n shots: {} +/- {}".format(np.sqrt(n_shots/5)*fit_hist_gauss(xdiffs5.x_l)[0][1], np.sqrt(n_shots/5)*fit_hist_gauss(xdiffs5.x_l)[1][1]),

    fig_name = "pointing_figure"
    rcParams.update({})
    scl = 2/1.61
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

    init_plt()

    gs0_bounds = {'wspace': 0.05}
    gs0 = gridspec.GridSpec(3, 5, wspace=.1, hspace=.1)
    gs0.update(**gs0_bounds)

    fig = pl.figure()

    '''
    Affine calibration
    '''
    ax0 = plt.subplot(gs0[1, 0:2])
    symbols = ['1', '2', '3', '4', '1', '2', '3', '4', '1', '2', '3']# 's', 'p', 'h', '+', 'x', 'D', 'o']
    red_mkr = mlines.Line2D([], [], color=color_or_rd, marker='o', markersize=3, label='atoms', linewidth=0)
    gray_mkr = mlines.Line2D([], [], color=color_grey, marker='o', markersize=3, label='camera', linewidth=0)
    for i, sym in zip(range(len(calibration_data.index)), symbols):
        ax0.plot(calibration_data.x_t[i] - mean_xt, calibration_data.x_l[i] - mean_xl, marker=sym, color=color_or_rd)
        ax0.plot(calibration_data.x_ct[i] - mean_xt, calibration_data.x_cl[i] - mean_xl, marker=sym, color=color_grey)

    ax0.legend(handles=[red_mkr, gray_mkr], numpoints=1, loc=3, fontsize=8, labelspacing=0, handletextpad=0.0)
    ax0.set_xlim([-5, 5])
    ax0.set_ylim([-6, 4])
    ax0.set_xlabel('$x_t$ (sites)')
    ax0.set_ylabel('$x_l$ (sites)')
    ax0.set_aspect('equal')

    xy = (-.25, 1.05)
    ax0.annotate('(c)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    '''
    Prediction vs measurement
    '''
    ax5 = plt.subplot(gs0[2, 0:2])
    ax5.plot(oos_data.x_t - mean_xt, oos_data.x_ct - mean_xt, color=cb.get_map('BuGn', 'Sequential', 9).mpl_colors[-3],
             linewidth=0, marker='o', ms=2, label='x_t')
    ax5.plot(oos_data.x_l - mean_xl, oos_data.x_cl - mean_xl, color=cb.get_map('PuBu', 'Sequential', 9).mpl_colors[-3],
             linewidth=0, marker='o', ms=2, label='x_l')
    ax5.plot([-10, 10], [-10, 10], linewidth=1, color=color_black)
    ax5.set_xlim([-2, 2])
    ax5.set_ylim([-2, 2])
    ax5.set_xlabel('$x$ (sites)')
    ax5.set_ylabel('$x_c$ (sites)')
    ax5.set_aspect('equal')
    ax5.xaxis.set_ticks(np.linspace(-2, 2, 5))
    ax5.yaxis.set_ticks(np.linspace(-2, 2, 5))
    ax5.legend(loc=2, numpoints=1, handletextpad=0.0)

    xy = (-.25, 1.1)
    ax5.annotate('(d)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    '''
    Data scan
    '''
    ax2 = plt.subplot(gs0[1, 2:])
    ax3 = plt.subplot(gs0[2, 2:])
    ax2.plot(oos_data.x_t - mean_xt, color=color_or_rd, label='atoms')
    ax2.plot(oos_data.x_ct - mean_xt, color=color_grey, label='camera')
    ax2.set_ylabel(r'$x_t$')
    ax2.set_ylim([-3, 3])
    ax2.legend(loc=1)
    plt.setp(ax2.get_xticklabels(), visible=False)

    ax3.plot(oos_data.x_l - mean_xl, color=color_or_rd, label='atoms')
    ax3.plot(oos_data.x_cl - mean_xl, color=color_grey, label='camera')
    ax3.set_ylabel(r'$x_l$')
    ax3.set_ylim([-3, 3])
    ax3.set_xlabel('Experimental realization index')
    ax3.legend(loc=2)
    xy = (-.2, 1.02)
    ax2.annotate('(e)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)
    ax3.annotate('(f)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)


    '''
    Histogram
    '''
    ax4 = plt.subplot(gs0[0, 0:3])
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



    ax4.legend(loc=2)

    ax4.set_ylabel(r'$P(\delta x)$')
    ax4.set_xlabel(r'$\delta x$')

    ax4.set_xlim([-3, 3])
    xy = (-.2, 1.02)
    ax4.annotate('(a)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    '''
    Autocorrelation
    '''
    autocorr_keys = ['x_t', 'x_l']
    colors = [cb.get_map('BuGn', 'Sequential', 9).mpl_colors[-3], cb.get_map('PuBu', 'Sequential', 9).mpl_colors[-3]]
    extent = 1, 10
    corrs = [(k, np.arange(*extent), np.array([xdiffs[k].autocorr(lag=n) for n in range(*extent)]), c) for k, c in zip(autocorr_keys, colors)]

    ax6 = plt.subplot(gs0[0, 3:])
    for corr in corrs:
        ax6.plot(corr[1], corr[2], label='${}$'.format(corr[0]), marker='o', color=corr[3])
    ax6.axhline(0, color='k')#, xmin=extent[0], xmax=extent[1], color=color_black)
    ax6.set_xlabel('lag (experiment cycles)')
    ax6.set_ylabel('autocorrelation')
    ax6.legend(loc=4, numpoints=1, handletextpad=0.3)
    xy = (-.3, 1.02)
    ax6.annotate('(b)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    gs0.tight_layout(fig)

    #xy = (-.3, 1.02)

    plt.savefig('{0}.pdf'.format(fig_name))