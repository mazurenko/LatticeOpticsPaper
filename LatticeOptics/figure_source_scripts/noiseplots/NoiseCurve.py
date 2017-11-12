'''
Created on Jan 22, 2016

@author: anton
'''
import numpy as np
import os

import matplotlib.pyplot as plt
import brewer2mpl as cb
from matplotlib.backends.backend_pdf import PdfPages
from plot_tools import *
from matplotlib import rcParams
import pylab as pl


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def sample_cmap(cmap, n_pts, bottom=0.0, top=1.0, gamma=1):
    """
    Produces n_pts colors sampled from the given cmap
    :param cmap: colormap to be sampled from. Assume it is a function on [0, 1] that maps to a color tuple
    :param n_pts: number of points to sample
    :param bottom: minimum input
    :param top: maximum input
    :param gamma: corresponds to nonlinearity
    :return: list of color tuples sampled from cmap
    """
    assert top <= 1
    assert bottom >= 0
    assert top >= bottom
    rng = float(top) - float(bottom)
    return map(cmap, bottom + rng*np.power(np.linspace(0, 1, n_pts), 1))


def make_rin_plot(nc_dict, x_window=[.8e3, 3e6], y_window=[-155,-85],
                  x_label='Frequency (Hz)', y_label="Rin (dBc/Hz)",
                  colors=cb.get_map('Set1', 'Qualitative', 4, reverse=True).mpl_colors, linewidth=1,
                  is_save=False, save_name='RIN', save_dir='plots', is_show=False):
    """
    :param nc_dict: dictionary of noise curves
    :param x_window: window of x axis
    :param y_window: window of y axis
    :param x_label: label of x
    :param y_label: label of y
    :param colors:
    :param linewidth:
    :param is_save:
    :param save_name:
    :param save_dir:
    :param is_show
    :return:
    """

    #settings
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

    for idx, (label, nc) in enumerate(nc_dict.iteritems()):
        print idx
        ax.plot(nc.mat[:, 0], nc.mat[:, 1], linewidth=linewidth, color=colors[idx], label=label, zorder=idx)

    ax.set_xlim(x_window)
    ax.set_ylim(y_window)
    ax.grid(zorder=-1, color=color_grey)

    ax.legend()

    if x_label is not None:
        ax.set_xlabel(x_label)
    if y_label is not None:
        ax.set_ylabel(y_label)

    ax.set_xscale('log')
    # fig.tight_layout()

    if is_save:
        out_name = '%s\\%s' % (save_dir, save_name)

        pp = PdfPages('%s.pdf' % out_name)
        plt.savefig(pp, format='pdf')
        pp.close()

    if is_show:
        plt.show()
    return ax


class NoiseCurve(object):
    def __init__(self, data_dir, pd_dc_voltage, lo_filename=None, med_filename=None, hi_filename=None,
                 optical_power = None, lo_rbw=62.5, med_rbw=500, hi_rbw=10, is_freq_normalized_SA=False,
                 is_freq_normalized_FFTM=False, offset=(0, 0, 0)):
        self.data_dir = data_dir
        self.lo_filename = "%s\\%s"%(data_dir, lo_filename) if lo_filename is not None else None
        self.med_filename = "%s\\%s"%(data_dir, med_filename) if med_filename is not None else None
        self.hi_filename = "%s\\%s"%(data_dir, hi_filename) if hi_filename is not None else None
        self.pd_dc_voltage = pd_dc_voltage
        self.pd_dc_power = np.square(self.pd_dc_voltage/2.0)/50 # the factor of 2 is for impedance, the factor of 50 is ohms
        
        self.lo_rbw = lo_rbw
        self.med_rbw = med_rbw
        self.hi_rbw = hi_rbw
        self.is_freq_normalized_SA=is_freq_normalized_SA
        self.is_freq_normalized_FFTM=is_freq_normalized_FFTM
        self.offset = offset
        
        self.lo_mat = None
        self.med_mat = None
        self.hi_mat = None
        self.mat = None
        self.load_traces()
        self.concatenate_traces()

    
    @staticmethod
    def parse_fft_machine_trace(filename):
        mat = np.loadtxt(filename, skiprows = 2)
        f = mat[:, 1]
        dB = mat[:, 0]
        return np.transpose(np.vstack((f, dB)))

    @staticmethod
    def parse_spectrum_analyzer_csv(filename):
        mat = np.genfromtxt(filename, skip_header=226, skip_footer=4, delimiter=',', usecols=(1, 2))
        f = mat[:, 1]*1e6
        dB = mat[:, 0]
        return np.transpose(np.vstack((f, dB)))
    
    def load_traces(self):
        if self.lo_filename is not None:
            self.lo_mat = self.parse_fft_machine_trace(self.lo_filename)
            if self.is_freq_normalized_FFTM:
                self.lo_mat[:, 1] = self.lo_mat[:, 1] - 10*np.log10(4) + 10*np.log10(20) - 10*np.log10(1e3 * self.pd_dc_power) + self.offset[0]
            else:
                self.lo_mat[:, 1] = self.lo_mat[:, 1] - 10*np.log10(4) + 10*np.log10(20) - 10*np.log10(self.lo_rbw) - 10*np.log10(1e3 * self.pd_dc_power) + self.offset[0]

        if self.med_filename is not None:
            self.med_mat = self.parse_fft_machine_trace(self.med_filename)

            if self.is_freq_normalized_FFTM:
                self.med_mat[:, 1] = self.med_mat[:, 1] - 10*np.log10(4) + 10*np.log10(20) - 10*np.log10(1e3 * self.pd_dc_power) + self.offset[1]
            else:
                self.med_mat[:, 1] = self.med_mat[:, 1] - 10*np.log10(4) + 10*np.log10(20) - 10*np.log10(self.med_rbw) - 10*np.log10(1e3 * self.pd_dc_power) + self.offset[1]

        if self.hi_filename is not None:
            self.hi_mat = self.parse_spectrum_analyzer_csv(self.hi_filename)
            if self.is_freq_normalized_SA:
                self.hi_mat[:, 1] = self.hi_mat[:, 1] - 10*np.log10(1e3 * self.pd_dc_power) + self.offset[2]
            else:
                self.hi_mat[:, 1] = self.hi_mat[:, 1] - 10 *np.log10(self.hi_rbw) - 10*np.log10(1e3 * self.pd_dc_power) + self.offset[2]

    def concatenate_traces(self):
        if self.lo_mat is not None and self.med_mat is not None:
            max_f_lo = np.max(self.lo_mat[:,0])
            med_start, nearest_element = find_nearest(self.med_mat[:,0], max_f_lo)
            self.mat = np.vstack((self.lo_mat, self.med_mat[med_start:-1,:]))

            if self.hi_mat is not None:
                max_f_med = np.max(self.med_mat[:,0])
                hi_start, nearest_element = find_nearest(self.hi_mat[:,0], max_f_med)
                self.mat = np.vstack((self.mat, self.hi_mat[hi_start:-1,:]))


if __name__ == '__main__':
    pass