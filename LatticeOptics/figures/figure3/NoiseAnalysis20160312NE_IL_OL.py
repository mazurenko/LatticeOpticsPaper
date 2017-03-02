'''
Created on Jan 22, 2016

@author: anton
'''


"""
This script outputs the noise curves for the 80048 nufern fiber amplifier in and out of loop, within the system and
at high power
"""

from collections import OrderedDict
from NoiseCurve import *
import brewer2mpl as cb
import numpy as np
import cmocean


if __name__ == '__main__':
    print cmocean.cm.haline

    # In loop
    noise_curves_80048_IL = OrderedDict()
    noise_curves_80048_IL['50 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename='50mW_lo.txt', med_filename='50mW_med.txt', hi_filename='measurement(174).csv', hi_rbw = 100)
    noise_curves_80048_IL['100 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename='100mW_lo.txt', med_filename='100mW_med.txt', hi_filename='measurement(176).csv', hi_rbw = 100)
    noise_curves_80048_IL['200 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename='200mW_lo.txt', med_filename='200mW_med.txt', hi_filename='measurement(178).csv', hi_rbw = 100)
    noise_curves_80048_IL['400 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename='400mW_lo.txt', med_filename='400mW_med.txt', hi_filename='measurement(180).csv', hi_rbw = 100)
    noise_curves_80048_IL['800 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename='800mW_lo.txt', med_filename='800mW_med.txt', hi_filename='measurement(182).csv', hi_rbw = 100)

    colors = sample_cmap(cmocean.cm.haline, 5, gamma=1, top=0.75)
    make_rin_plot(noise_curves_80048_IL, is_save=True, save_name='80048_NE_IL', colors=colors)

    # Out of loop
    noise_curves_80048_OL = OrderedDict()
    noise_curves_80048_OL['50 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '50mW_lo_ol.txt', med_filename='50mW_med_ol.txt', hi_filename='measurement(175).csv', hi_rbw = 100)
    noise_curves_80048_OL['100 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '100mW_lo_ol.txt', med_filename='100mW_med_ol.txt', hi_filename='measurement(177).csv', hi_rbw = 100)
    noise_curves_80048_OL['200 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '200mW_lo_ol.txt', med_filename='200mW_med_ol.txt', hi_filename='measurement(179).csv', hi_rbw = 100)
    noise_curves_80048_OL['400 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '400mW_lo_ol.txt', med_filename='400mW_med_ol.txt', hi_filename='measurement(181).csv', hi_rbw = 100)
    noise_curves_80048_OL['800 mW'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '800mW_lo_ol.txt', med_filename='800mW_med_ol.txt', hi_filename='measurement(183).csv', hi_rbw = 100)

    make_rin_plot(noise_curves_80048_OL, is_save=True, save_name='80048_NE_OL', colors=colors)

    # High power in/out of loop
    noise_curves_80048_hp = OrderedDict()
    noise_curves_80048_hp['2 W, IL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '2W_lo.txt', med_filename='2W_med.txt', hi_filename='measurement(185).csv',hi_rbw = 100)
    noise_curves_80048_hp['2 W, OL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '2W_lo_ol.txt', med_filename='2W_med_ol.txt', hi_filename='measurement(184).csv',hi_rbw = 100)

    hp_colors = sample_cmap(cmocean.cm.algae, 2, top=0.8, bottom=0.2)
    make_rin_plot(noise_curves_80048_hp, is_save=True, save_name='80048_NE_hp', colors=hp_colors)

