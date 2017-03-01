'''
Created on Jan 22, 2016

@author: anton
'''
import numpy as np

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


class NoiseCurve(object):
    def __init__(self,data_dir, pd_dc_voltage, lo_filename = None, med_filename = None, hi_filename = None,optical_power = None,lo_rbw = 62.5, med_rbw = 500, hi_rbw = 10):
        self.data_dir = data_dir
        self.lo_filename = "%s\\%s"%(data_dir, lo_filename) if lo_filename is not None else None
        self.med_filename = "%s\\%s"%(data_dir, med_filename) if med_filename is not None else None
        self.hi_filename = "%s\\%s"%(data_dir, hi_filename) if hi_filename is not None else None
        self.pd_dc_voltage = pd_dc_voltage
        self.pd_dc_power = np.square(self.pd_dc_voltage/2.0)/50 # the factor of 2 is for impedance, the factor of 50 is ohms
        
        self.lo_rbw = lo_rbw
        self.med_rbw = med_rbw
        self.hi_rbw = hi_rbw
        
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
        mat = np.genfromtxt(filename, skip_header = 226, skip_footer = 4, delimiter = ',', usecols = (1,2))
        f = mat[:,1]*1e6
        dB = mat[:,0]
        return np.transpose(np.vstack((f,dB)))
    
    def load_traces(self):
        if self.lo_filename is not None:
            self.lo_mat = self.parse_fft_machine_trace(self.lo_filename)
            self.lo_mat[:,1] = self.lo_mat[:,1] - 10*np.log10(4) + 10*np.log10(20) - 10*np.log10(self.lo_rbw) - 10*np.log10(1e3 * self.pd_dc_power)

        if self.med_filename is not None:
            self.med_mat = self.parse_fft_machine_trace(self.med_filename)
            self.med_mat[:,1] = self.med_mat[:,1] - 10 *np.log10(4) + 10*np.log10(20) -10*np.log10(self.med_rbw) - 10*np.log10(1e3 * self.pd_dc_power)

        if self.hi_filename is not None:
            self.hi_mat = self.parse_spectrum_analyzer_csv(self.hi_filename)
            self.hi_mat[:,1] = self.hi_mat[:,1] - 10 *np.log10(self.hi_rbw) - 10*np.log10(1e3 * self.pd_dc_power)
    
    def concatenate_traces(self):
        if self.lo_mat is not None and self.med_mat is not None:
            max_f_lo = np.max(self.lo_mat[:,0])
            med_start, nearest_element = find_nearest(self.med_mat[:,0], max_f_lo)
            #print np.shape(self.lo_mat)
            #print np.shape(self.med_mat)
            print med_start
            self.mat = np.vstack((self.lo_mat, self.med_mat[med_start:-1,:]))

            if self.hi_mat is not None:
                max_f_med = np.max(self.med_mat[:,0])
                hi_start, nearest_element = find_nearest(self.hi_mat[:,0], max_f_med)
                self.mat = np.vstack((self.mat, self.hi_mat[hi_start:-1,:]))

        '''
        if self.lo_mat is not None:
            self.mat = self.lo_mat
        '''


if __name__ == '__main__':
    pass