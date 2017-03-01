'''
Created on Jan 22, 2016

@author: anton
'''
import matplotlib.pyplot as plt
from NoiseCurve import NoiseCurve
import brewer2mpl as cb
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np
import scipy.integrate as integrate
from scipy.interpolate import interp1d
        

def ensure_dir(f):
    print f
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)


def make_rin_plot(nc_dict, x_window = [1e3,5e6], y_window = [-150,-80], 
                  x_label = 'Frequency (Hz)', y_label = "Rin (dBc/Hz)",
                  colors = cb.get_map('Set3', 'Qualitative', 10 , reverse=True).mpl_colors,
                  is_save = False, save_name = 'RIN', save_dir = 'plots', fig = None, ax = None):
    '''
    :param nc_dict dictionary of descriptive strings to noise curves
    
    makes a pretty rin plot
    '''
    #settings
    linewidth = 3
    
    
    #make plot
    if fig is None:
        fig = plt.figure()
    if ax is None:
        ax = fig.add_subplot(1,1,1)

    idx = 0
    for label, nc in nc_dict.iteritems():
        ax.plot(nc.mat[:,0],nc.mat[:,1], linewidth = linewidth,color = colors[idx],label = label)
        idx = idx + 1
    
    ax.set_xlim(x_window)
    ax.set_ylim(y_window)
    ax.grid()
    
    ax.legend()

    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlabel(x_label,fontsize=20)
    ax.set_ylabel(y_label,fontsize=20)

    ax.set_xscale('log')
    fig.tight_layout()
    
    if is_save:
        #ensure_dir(save_dir)
        out_name = '%s\\%s'%(save_dir,save_name)

        pp=PdfPages('%s.pdf'%out_name)
        plt.savefig(pp,format='pdf')
        pp.close()
        plt.savefig('%s.png'%out_name)

        
    plt.show()


if __name__ == '__main__':


    noise_curves_80048_IL = {}

    noise_curves_80048_IL['50 mW, IL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '50mW_lo.txt', med_filename='50mW_med.txt', hi_filename='measurement(174).csv',hi_rbw = 100)
    print noise_curves_80048_IL['50 mW, IL'].mat

    noise_curves_80048_IL['50 mW, OL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '50mW_lo_ol.txt', med_filename='50mW_med_ol.txt', hi_filename='measurement(175).csv',hi_rbw = 100)

    noise_curves_80048_IL['100 mW, IL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '100mW_lo.txt', med_filename='100mW_med.txt', hi_filename='measurement(176).csv',hi_rbw = 100)
    noise_curves_80048_IL['100 mW, OL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '100mW_lo_ol.txt', med_filename='100mW_med_ol.txt', hi_filename='measurement(177).csv',hi_rbw = 100)

    noise_curves_80048_IL['200 mW, IL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '200mW_lo.txt', med_filename='200mW_med.txt', hi_filename='measurement(178).csv',hi_rbw = 100)
    noise_curves_80048_IL['200 mW, OL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '200mW_lo_ol.txt', med_filename='200mW_med_ol.txt', hi_filename='measurement(179).csv',hi_rbw = 100)

    noise_curves_80048_IL['400 mW, IL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '400mW_lo.txt', med_filename='400mW_med.txt', hi_filename='measurement(180).csv',hi_rbw = 100)
    noise_curves_80048_IL['400 mW, OL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '400mW_lo_ol.txt', med_filename='400mW_med_ol.txt', hi_filename='measurement(181).csv',hi_rbw = 100)

    noise_curves_80048_IL['800 mW, IL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '800mW_lo.txt', med_filename='800mW_med.txt', hi_filename='measurement(182).csv',hi_rbw = 100)
    noise_curves_80048_IL['800 mW, OL'] = NoiseCurve('20160312_NE_RIN_80048\\data', 2.00, lo_filename= '800mW_lo_ol.txt', med_filename='800mW_med_ol.txt', hi_filename='measurement(183).csv',hi_rbw = 100)

    for nc in noise_curves_80048_IL.values():
        nc.load_traces()
        nc.concatenate_traces()
    
    make_rin_plot(noise_curves_80048_IL,is_save=True,save_name = '80048_NE')

    noise_curves_80048_hp = {}
    noise_curves_80048_hp['2 W, IL'] = NoiseCurve('20160312_NE_RIN_80048\\data',2.00, lo_filename= '2W_lo.txt', med_filename='2W_med.txt', hi_filename='measurement(185).csv',hi_rbw = 100)
    noise_curves_80048_hp['2 W, OL'] = NoiseCurve('20160312_NE_RIN_80048\\data',2.00, lo_filename= '2W_lo_ol.txt', med_filename='2W_med_ol.txt', hi_filename='measurement(184).csv',hi_rbw = 100)

    for nc in noise_curves_80048_hp.values():
        nc.load_traces()
        nc.concatenate_traces()
    
    make_rin_plot(noise_curves_80048_IL,is_save=True,save_name = '80048_NE')
    make_rin_plot(noise_curves_80048_hp,is_save=True,save_name = '80048_NE_hp')



    '''

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)

    #ax.plot(nc_bg.mat[:,0],nc_bg.mat[:,1], nc_me.mat[:,0],nc_me.mat[:,1])
    for nc in noise_curves_8971.values():
        ax.plot(nc.mat[:,0],nc.mat[:,1])

    ax.set_xscale('log')


    #nc_bg = NoiseCurve('20160122_RIN',lo_filename = 'background12khz.txt')
    #nc_bg.load_traces()
    #plt.plot(nc_bg.lo_mat[:,0],nc_bg.lo_mat[:,1])
    plt.show()
    
    '''