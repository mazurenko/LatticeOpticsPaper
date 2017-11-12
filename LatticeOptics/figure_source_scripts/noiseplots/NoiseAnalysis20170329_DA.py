'''
Created on Jan 22, 2016

@author: anton
'''
from NoiseCurve import *
import numpy as np
import cmocean

from collections import OrderedDict
        

if __name__ == '__main__':

    colors = cb.get_map('OrRd', 'Sequential', 6, reverse=False).mpl_colors[2:]  # Nufern colors

    noise_curves_DA = OrderedDict()
    noise_curves_DA['fixed'] = NoiseCurve('20170329_DA_noise', 1.95, lo_filename='DA_fixed_lo.txt',
                                            med_filename='DA_fixed_med.txt', hi_filename='fixed_spectrum.csv',
                                            hi_rbw=100, is_freq_normalized_FFTM=True)

    noise_curves_DA_scanning = OrderedDict()
    #noise_curves_DA_scanning['DA'] = NoiseCurve('20170329_DA_noise', 1.95, lo_filename='DA_scan_lo.txt',
    #                                        med_filename='DA_scan_med.txt', hi_filename='sawtooth_spectrum.csv',
    #                                        hi_rbw=100, is_freq_normalized_FFTM=True, offset=(0, 0, 40))

    noise_curves_DA_scanning['fixed'] = NoiseCurve('20170329_DA_noise', 1.95, lo_filename='DA_fixed_lo.txt',
                                            med_filename='DA_fixed_med.txt', hi_filename='fixed_spectrum.csv',
                                            hi_rbw=100, is_freq_normalized_FFTM=True)

    noise_curves_DA_scanning['scanning'] = NoiseCurve('20170414_DA_Scan', 1.75, lo_filename='DA_20ms_ramp_lo.txt',
                                                med_filename='DA_20ms_ramp_med.txt', hi_filename='measurement(210).csv',
                                                hi_rbw=100, is_freq_normalized_FFTM=True)

    colors = sample_cmap(cmocean.cm.haline, 2, top=0.6, bottom=0.0)
    ax = make_rin_plot(noise_curves_DA_scanning, is_save=True, save_name='da_rin_scan', x_window=[.8e3, 10e6],
                  y_window=[-165, -90], colors = colors)

