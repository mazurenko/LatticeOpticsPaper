'''
Created on Jan 22, 2016

@author: anton
'''
from NoiseCurve import *
import numpy as np

from collections import OrderedDict
        

if __name__ == '__main__':
    
    noise_curves_bg = OrderedDict()
    noise_curves_bg['background'] = NoiseCurve('20160122_RIN', 2.0, lo_filename='background12khz.txt',
                                               med_filename='background100khz.txt', hi_filename='measurement(144).csv')
    noise_curves_bg['mephisto'] = NoiseCurve('20160122_RIN', 2.0, lo_filename='mephisto12khz.txt',
                                             med_filename='mephisto100khz.txt', hi_filename='measurement(145).csv')
    
    noise_curves_80048 = OrderedDict()
    noise_curves_80048['33.7A'] = NoiseCurve('20160122_RIN\\80048', 2.0, lo_filename='33p7_12kHz.txt',
                                             med_filename='33p7_100kHz.txt', hi_filename='measurement(150).csv',
                                             hi_rbw=100)
    noise_curves_80048['30A'] = NoiseCurve('20160122_RIN\\80048', 2.0, lo_filename='30_12kHz.txt',
                                           med_filename='30_100kHz.txt', hi_filename='measurement(153).csv',
                                           hi_rbw=100)
    noise_curves_80048['25A'] = NoiseCurve('20160122_RIN\\80048', 2.0, lo_filename='25_12kHz.txt',
                                           med_filename='25_100kHz.txt', hi_filename='measurement(151).csv', hi_rbw=100)
    noise_curves_80048['20A'] = NoiseCurve('20160122_RIN\\80048', 2.0, lo_filename='20_12kHz.txt',
                                           med_filename='20_100kHz.txt', hi_filename='measurement(154).csv', hi_rbw=100)
    noise_curves_80048['Seed'] = NoiseCurve('20160122_RIN', 2.0, lo_filename='mephisto12khz.txt',
                                            med_filename='mephisto100khz.txt', hi_filename='measurement(145).csv')
    """
    noise_curves_80048['Detector'] = NoiseCurve('20160122_RIN', 2.0, lo_filename='background12khz.txt',
                                                  med_filename='background100khz.txt',
                                                  hi_filename='measurement(144).csv')
    """

    colors = cb.get_map('OrRd', 'Sequential', 6, reverse=False).mpl_colors[2:]  # Nufern colors
    colors.extend([cb.get_map('YlGn', 'Sequential', 3, reverse=False).mpl_colors[-1]])  # seed
    colors.extend([cb.get_map('YlGnBu', 'Sequential', 3, reverse=False).mpl_colors[-1]])  # detector

    make_rin_plot(noise_curves_80048, is_save=True, save_name='80048_unmodified', colors=colors)


    '''
    noise_curves_8971 = OrderedDict()
    noise_curves_8971['43.5A'] = NoiseCurve('20160122_RIN\\8971', 2.0, lo_filename='43p5A12kHz.txt',
                                            med_filename='43p5A100kHz.txt', hi_filename='measurement(148).csv',
                                            hi_rbw=100)
    noise_curves_8971['30A'] = NoiseCurve('20160122_RIN\\8971', 2.0, lo_filename='30A12kHz.txt',
                                          med_filename='30A100kHz.txt', hi_filename='measurement(147).csv', hi_rbw=100)
    noise_curves_8971['20A'] = NoiseCurve('20160122_RIN\\8971', 2.0, lo_filename='20A12kHz.txt',
                                          med_filename='30A100kHz.txt', hi_filename='measurement(149).csv', hi_rbw=100)


    make_rin_plot(noise_curves_8971, is_save=True, save_name='8971_20160122')
    '''
