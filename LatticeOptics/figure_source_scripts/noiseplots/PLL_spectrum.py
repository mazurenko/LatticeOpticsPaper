import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pylab as pl
from plot_tools import *
import matplotlib as mpl
import brewer2mpl as cb
from matplotlib import rcParams
import brewer2mpl as cb
import cmocean as cm

if __name__ == "__main__":
    data_dir = 'pll_spectrum'
    plot_dir = 'plots'
    data_file = 'freq_spectrum.csv'

    spectrum = np.genfromtxt('{}//{}'.format(data_dir, data_file), delimiter=',', skip_footer=6, skip_header=226)[:, 1:3]
    print spectrum

    fig_name = "pll_spectrum"
    rcParams.update({})
    scl = 1/1.61
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
    ax = fig.add_subplot(1, 1, 1)
    colors_val_cb = cb.get_map('RdGy', 'diverging', 5).mpl_colors
    ax.grid(zorder=0, color=color_grey)
    ax.plot(spectrum[:, 1], spectrum[:, 0]-np.max(spectrum[:, 0]), color=colors_val_cb[0], zorder=1)
    ax.set_xlabel('Frequency (MHz)')
    ax.set_ylabel('Power (dBc)')
    ax.set_xlim([77.75, 82.25])
    ax.set_ylim([-105, 5])
    ax.legend()

    plt.savefig('{0}//{1}.pdf'.format(plot_dir, fig_name))