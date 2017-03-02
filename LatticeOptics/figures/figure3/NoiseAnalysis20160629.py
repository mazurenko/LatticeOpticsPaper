'''
Created on Jan 22, 2016

@author: anton
'''
from NoiseCurve import *
import collections
import numpy as np
from collections import OrderedDict
import cmocean

"""
README:
WE VARIED THE CURRENTS OF NW LATTICE 9213, in an effort to reduce the RIN - did not make huge difference
We measured the RIN in closed loop at 100, 200 and 400 mW, always at 2 V offset voltage - consistent with NE
We measured the RIN in closed loop, with at 2 kHz filter in addition to the cutoff filter.
"""

if __name__ == '__main__':

    #CLOSED LOOP MEASUREMENT, with LOW PASS
    noise_curves_9213_cl_LP = OrderedDict()
    noise_curves_9213_cl_LP['100mW'] = NoiseCurve('20160629_RIN\\data', 2, lo_filename= '100mW_lo_LP.txt', med_filename='100mW_med_LP.txt', hi_filename='measurement(199).csv',hi_rbw = 100)
    noise_curves_9213_cl_LP['200mW'] = NoiseCurve('20160629_RIN\\data', 2, lo_filename= '200mW_lo_LP.txt', med_filename='200mW_med_LP.txt', hi_filename='measurement(198).csv',hi_rbw = 100)
    noise_curves_9213_cl_LP['400mW'] = NoiseCurve('20160629_RIN\\data', 2, lo_filename= '400mW_lo_LP.txt', med_filename='400mW_med_LP.txt', hi_filename='measurement(197).csv',hi_rbw = 100)

    lp_colors = sample_cmap(cmocean.cm.speed, 3, bottom=0.3)[::-1]
    make_rin_plot(noise_curves_9213_cl_LP, is_save=True, save_name='9213_with_lp', colors=lp_colors)

    #CLOSED LOOP MEASUREMENT
    noise_curves_9213_cl = collections.OrderedDict()
    noise_curves_9213_cl['100mW'] = NoiseCurve('20160629_RIN\\data',2, lo_filename= '100mW_lo.txt', med_filename='100mW_med.txt', hi_filename='measurement(194).csv',hi_rbw = 100)
    noise_curves_9213_cl['200mW'] = NoiseCurve('20160629_RIN\\data',2, lo_filename= '200mW_lo.txt', med_filename='200mW_med.txt', hi_filename='measurement(195).csv',hi_rbw = 100)
    noise_curves_9213_cl['400mW'] = NoiseCurve('20160629_RIN\\data',2, lo_filename= '400mW_lo.txt', med_filename='400mW_med.txt', hi_filename='measurement(196).csv',hi_rbw = 100)

    no_lp_colors = lp_colors
    make_rin_plot(noise_curves_9213_cl, is_save=True, save_name='9213_without_lp', colors=no_lp_colors)


    noise_curves_9213 = {}
    noise_curves_9213['NW_37A'] = NoiseCurve('20160629_RIN\\data',2, lo_filename= '37A_railed_lo.txt', med_filename='37A_railed_med.txt', hi_filename='measurement(191).csv',hi_rbw = 100)
    noise_curves_9213['NW_35A'] = NoiseCurve('20160629_RIN\\data',2, lo_filename= '35A_railed_lo.txt', med_filename='35A_railed_med.txt', hi_filename='measurement(192).csv',hi_rbw = 100)
    noise_curves_9213['NW_32A'] = NoiseCurve('20160629_RIN\\data',2, lo_filename= '32A_railed_lo.txt', med_filename='32A_railed_med.txt', hi_filename='measurement(193).csv',hi_rbw = 100)

    #make_rin_plot(noise_curves_9213, is_save=True, save_name='9213_Currents_20160629')

