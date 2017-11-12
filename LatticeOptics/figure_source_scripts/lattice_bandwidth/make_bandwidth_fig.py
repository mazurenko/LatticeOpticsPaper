import numpy as np
import scipy as sp
import pandas as pd
import os
import matplotlib.pyplot as plt
import pylab as pl
from plot_tools import *
import matplotlib as mpl
import brewer2mpl as cb
from matplotlib import rcParams
import brewer2mpl as cb
import cmocean as cm
import matplotlib.gridspec as gridspec

def plot_errorbar(ax, df, columns, outer_colors, inner_colors, labels=None,
                  xlabel=None, ylabel=None, xlim=None, ylim=None):
    """
    plots the selected columns of df against their index on ax
    :param ax:
    :param df:
    :param colunmns:
    :param colors:
    :param light_fac: lightness factor
    :return:
    """

    for idx, column in enumerate(columns):
        cd = outer_colors[idx]
        cl = inner_colors[idx]
        #cl = tuple([(x+light_fac)/(1+light_fac) for x in list(cd)])
        #cl = tuple([(x+light_fac)/(1+light_fac) for x in list(cd)])

        pars = {'linewidth': 1,
                'fmt': 'o',
                'ms': 3,  # marker size
                'mew': 1,  # marker edge width
                'c': cl,
                'mec': cd,
                'mfc': cl,
                'capsize': 0,
                'zorder': .2*idx+3,
                'yerr': df[column+'_err'],
                'label': column if labels is None else labels[column]}

        ax.errorbar(df.index.values, df[column].values, **pars)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    legend = ax.legend(loc=3, frameon=True)
    legend.get_frame().set_linewidth(0)


    ax.grid(zorder=0, color=color_grey)

if __name__ == "__main__":
    data_ne = pd.read_csv(os.path.join('results', 'ne_results.csv'))
    data_ne = data_ne.set_index('power')
    data_ne = data_ne/1e3  # let's work in kHz, not Hz

    data_nw = pd.read_csv(os.path.join('results', 'nw_results.csv'))
    data_nw = data_nw.set_index('power')
    data_nw = data_nw/1e3  # let's work in kHz, not Hz

    label_dict = {'three_dB_pt': "3 dB point",
                  'pi_phase_point': r"$\pi$ phase point"}

    fig_name = "low_power_bandwidths"
    rcParams.update({})
    scl = 2/1.61
    fig_width = 3.37
    fig_height = scl*fig_width

    rcpars.update({
        "figure.autolayout": True,
        "figure.figsize": (fig_width, fig_height),
        "figure.subplot.left": .25,
        "figure.subplot.right": .94,
        "figure.subplot.bottom": .2,
        "figure.subplot.top": .95,
    })

    pl.rcParams.update(rcpars)

    init_plt()
    fig = pl.figure()
    ax0 = fig.add_subplot(2, 1, 1)
    ax1 = fig.add_subplot(2, 1, 2)

    outer_colors = [cb.get_map('OrRd', 'Sequential', 6, reverse=False).mpl_colors[-1]]
    outer_colors.extend([cb.get_map('YlGn', 'Sequential', 3, reverse=False).mpl_colors[-1]])

    inner_colors = [cb.get_map('OrRd', 'Sequential', 6, reverse=False).mpl_colors[-3]]
    inner_colors.extend([cb.get_map('YlGn', 'Sequential', 6, reverse=False).mpl_colors[-3]])

    plot_errorbar(ax0, data_ne, ['three_dB_pt', 'pi_phase_point'], outer_colors, inner_colors, labels=label_dict,
                  xlim=[-50, 1050], ylim=[-1, 15], ylabel='frequency (kHz)')
    #  remove tick labels in ax0
    labels = [item.get_text() for item in ax0.get_xticklabels()]
    empty_string_labels = [''] * len(labels)
    ax0.set_xticklabels(empty_string_labels)

    xy = (-.17, .97)
    ax0.annotate('(a)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    #  ax1
    plot_errorbar(ax1, data_nw, ['three_dB_pt', 'pi_phase_point'], outer_colors, inner_colors, labels=label_dict,
                  xlim=[-30, 1030], ylim=[-1, 15], xlabel='power (mW)', ylabel='frequency (kHz)')

    ax1.annotate('(b)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)
    plt.savefig('{0}.pdf'.format(fig_name))