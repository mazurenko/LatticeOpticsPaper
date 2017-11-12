import numpy as np
import pickle as pkl
import pyfits
import os
import re
import matplotlib.pyplot as plt

"""
the data was scans 16 (calibration) 17-19 (data) on April 3, 2017.
"""
def binarize(amp_mat, threshold=0.21):
    return (amp_mat > threshold).astype(np.float32)


def get_camera_pointing_data(scans, runlog_dir="W:\\RunLog", name="data", log_regex = 'log-0\.csv'):
    compiled_pointing = []
    for datevec in scans:
        beam_mon_data_path = '%s\\%04d\\%02d\\%02d\\%04d%02d%02d-%04d\\beam_mon' % (runlog_dir, datevec[0], datevec[1], datevec[2],
                                                                                datevec[0], datevec[1], datevec[2], datevec[3])#,
                                                                                #datevec[0], datevec[1], datevec[2], datevec[3])
        print os.listdir(beam_mon_data_path)
        data_list = filter(lambda x: re.match(log_regex, x), os.listdir(beam_mon_data_path))
        print data_list
        scan_measurements = []
        for pointing in data_list:
            pd = np.loadtxt(os.path.join(beam_mon_data_path, pointing), delimiter=',')
            scan_measurements.append(pd)
        compiled_pointing.append(reduce(lambda x, y: np.hstack((x, y)), scan_measurements))

    compiled_data = reduce(lambda x, y: np.vstack((x, y)), compiled_pointing)
    np.savetxt('{}.csv'.format(name), compiled_data, delimiter=',')
    return 0


def get_single_site_data(scans, runlog_dir="W:\\RunLog", name="data", threshold=0.19):
    """
    gets the site analyzed, binarized data from all the specified scans, dunps them into a fitfile
    :param scans: (year,month,day,scan tuple)
    :param runlog_dir:
    :return:
    """
    binary_occupations = [pyfits.PrimaryHDU(np.array(scans))]
    for datevec in scans:
        print datevec
        ss_data_path = '%s\\%04d\\%02d\\%02d\\%04d%02d%02d-%04d\\FitResults' % (runlog_dir, datevec[0], datevec[1], datevec[2],
                                                                                datevec[0], datevec[1], datevec[2], datevec[3])#,
                                                                                #datevec[0], datevec[1], datevec[2], datevec[3])
        ss_data_list = filter(lambda x: re.match('fitresult_[0-9]{3}-[0-9]{2}\.pkl', x), os.listdir(ss_data_path))
        for ss_fit in ss_data_list:
            with open(os.path.join(ss_data_path, ss_fit), mode='rb') as f:
                ss_data = binarize(pkl.load(f)['amps'], threshold=threshold)
                binary_occupations.append(pyfits.ImageHDU(ss_data))
    hdulist = pyfits.HDUList(binary_occupations)
    hdulist.writeto('{}.fits'.format(name))


def load_fits(filename):
    hdulist = pyfits.open(filename)
    for i in range(1, len(hdulist)):
        print hdulist[i].data
        plt.imshow(hdulist[i].data)
        plt.show()

if __name__ == "__main__":
    # new scan, after sufficient warmup
    calibration_scans = [(2017, 04, 07, 53)]
    stability_scans = [(2017, 04, 07, 54), (2017, 04, 07, 55)]
    get_single_site_data(calibration_scans, name='calibration_2', threshold=0.15)
    get_camera_pointing_data(calibration_scans, name='calibration_2')
    get_single_site_data(stability_scans, name='stability_check_2', threshold=0.14)
    get_camera_pointing_data(stability_scans, name='stability_check_2')


    '''
    #Initial dataset exhibiting prohibitive drift
    get_single_site_data([(2017, 04, 03, 16)], name='calibration')
    get_camera_pointing_data([(2017, 04, 03, 16)], name='calibration')
    #get_single_site_data([(2017, 04, 03, 17), (2017, 04, 03, 18), (2017, 04, 03, 19)], name='stability_check')
    get_camera_pointing_data([(2017, 04, 03, 17), (2017, 04, 03, 18), (2017, 04, 03, 19)], name='stability_check')
    '''


