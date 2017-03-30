'''
Created on Jan 22, 2016

@author: anton
'''
from NoiseCurve import *
import numpy as np

from collections import OrderedDict
        

if __name__ == '__main__':
    
    noise_curves_PLL = OrderedDict()
    noise_curves_PLL['noise floor'] = NoiseCurve('20170326_PLL_RIN', 0.240, lo_filename='noise_floor_lo.txt',
                                             med_filename='noise_floor_med.txt', hi_filename='noise_floor.csv',
                                             hi_rbw=100, is_freq_normalized_FFTM=True)
    noise_curves_PLL['PLL'] = NoiseCurve('20170326_PLL_RIN', 0.240, lo_filename='pll_lo_noamp.txt',
                                            med_filename='pll_med_noamp.txt', hi_filename='high_freq_RIN.csv',
                                            hi_rbw=100, is_freq_normalized_FFTM=True)

    colors = cb.get_map('OrRd', 'Sequential', 6, reverse=False).mpl_colors[2:]  # Nufern colors

    make_rin_plot(noise_curves_PLL, is_save=True, save_name='pll_rin', x_window=[.8e3, 10e6],
                  y_window=[-175,-120])

