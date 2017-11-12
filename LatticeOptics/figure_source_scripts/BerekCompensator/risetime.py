import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import pylab as pl
from plot_tools import *
import matplotlib as mpl
import brewer2mpl as cb
from matplotlib import rcParams
import brewer2mpl as cb
import cmocean as cm
import matplotlib.gridspec as gridspec
from bandwidth_analyzer import *

if __name__ == "__main__":
    fig_name = "step_response"
    step_data = pd.read_csv('data//berek_bw.csv')[3:]
    step_data = step_data.rename(columns={'x-axis':'t', '2':'v'})
    step_data = step_data.astype(float)
    print step_data[:100].v.mean()
    print step_data[-100:].v.mean()
    rcParams.update({})
    scl = 1/1.61
    fig_width = 5.5
    fig_height = scl*fig_width

    rcpars.update({
        "figure.autolayout": True,
        "figure.figsize": (fig_width, fig_height),
        "figure.subplot.left": .1,
        "figure.subplot.right": .94,
        "figure.subplot.bottom": .2,
        "figure.subplot.top": .95,
    })

    pl.rcParams.update(rcpars)

    init_plt()
    fig = pl.figure()
    gs0 = gridspec.GridSpec(2, 2, wspace=.1, hspace=.1)
    ax0 = plt.subplot(gs0[0:, 0])
    ax0.set_xlabel('time (ms)')
    ax0.set_ylabel('$P$ (arb)')
    ax0.set_xlim([-1, 2])
    ax0.set_ylim([0, .8])

    ax0.plot(1e3*step_data.t+.148, step_data.v, color=color_or_rd) # the .148 is to trigger from the fungen edge, not the arbitrary t=0
    ax0.grid(zorder=-1, color=color_grey)
    xy = (-.25, 1)
    ax0.annotate('(a)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    ax1 = plt.subplot(gs0[0, 1])#fig.add_subplot(3, 1, 2)

    step_data = pd.read_csv('data//berek_bw.csv')[3:]
    step_data = step_data.rename(columns={'x-axis': 't', '2': 'v'})
    step_data = step_data.astype(float)
    start = step_data[:100].v.mean()
    stop = step_data[-100:].v.mean()
    step_data.v -= start
    step_data.v /= (stop-start)

    fourier_transform = fourier_ser(pd.Series(step_data.v.values, index=step_data.t.values), time_shift=936*2.5e-6)
    ft_amp = fourier_transform.amp

    freq_win = [100, 10000]
    #plt.plot(ft_amp[freq_win[0]:freq_win[1]].index, ft_amp[freq_win[0]:freq_win[1]].index*np.abs(ft_amp[freq_win[0]:freq_win[1]]))

    """
    To get bode plot, assume that the near dc gain is 1, and normalize to that. Sadly the closest we get is about 100 Hz,
    but that's close enough for a reasonably good estimate
    """
    effective_dc_freq = 100
    effective_dc_loc = fourier_transform.index.get_loc(effective_dc_freq, 'ffill')
    transfer_function = fourier_transform
    transfer_function.amp = fourier_transform.amp*fourier_transform.index # compensate for the fourier transform of step response
    transfer_function.angle = (fourier_transform.angle + np.pi/2) % (2*np.pi) - 2*np.pi # compensate for fourier transform of step response
    #transfer_function.angle = ((fourier_transform.angle - np.pi/2) + np.pi/2) % np.pi - np.pi/2# compensate for the fourier transform of step response
    #transfer_function.angle = ((fourier_transform.angle - np.pi/2) + np.pi/2) % np.pi - np.pi/2# compensate for the fourier transform of step response
    print transfer_function.angle
    effective_dc_norm = transfer_function.amp.iloc[effective_dc_loc]
    transfer_function.amp /= effective_dc_norm
    #transfer_function.amp /= transfer_function.amp.loc(effective_dc_loc)
    transfer_function = transfer_function[freq_win[0]:freq_win[1]]

    ax1.semilogy(transfer_function.index, transfer_function.amp, color=color_black)
    ax1.set_ylim([.1, 3])
    ax1.set_xscale('log')
    ax1.grid(zorder=-1, color=color_grey, which='both')
    ax1.set_xlabel('$f$ (Hz)')
    ax1.set_ylabel('$G$')

    ax1.annotate('(b)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    ax2 = plt.subplot(gs0[1, 1])#fig.add_subplot(3, 1, 3)
    ax2.plot(transfer_function.index, transfer_function.angle/np.pi, color=color_black)
    ax2.set_xscale('log')
    ax2.grid(zorder=-1, color=color_grey, which='both')
    ax2.set_xlabel('$f$ (Hz)')
    ax2.set_ylabel('$\phi/\pi$ (rad)')

    ax2.annotate('(c)', xy=xy, xytext=xy, xycoords='axes fraction', fontsize=12)

    gs0.tight_layout(fig)
    plt.savefig('{0}.pdf'.format(fig_name))