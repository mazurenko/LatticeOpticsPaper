import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
import os
import re
from plastia.analysis import fit

DATA_DIR = 'data'


def find_files(dir, axis):
    """
    looks for all files in the format "img-[num]-[axis].png" in the dirname directoy under data
    returns dict of shot to filename
    """
    all_files = os.listdir(dir)
    regex = re.compile('img-([0-9]*)-{}\.png'.format(axis))
    file_ids = {}
    for f in all_files:
        m = regex.match(f)
        id = int(m.groups()[0]) if m else None
        if id:
            file_ids[id] = f
    return file_ids


def get_errorbars(dirname, axis=0):
    """
    looks for all files in the format "img-[num]-[axis].png" in the dirname directoy under data
    returns the fitted parameters in a dataframe
    :param dirname:
    :return:
    """
    dirname = os.path.join(DATA_DIR, dirname)
    file_ids = find_files(dirname, axis=axis)
    for shot_num, filename in file_ids.iteritems():
        im = misc.imread(os.path.join(dirname, filename))
        f = fit.FitFunctionGaussian2D()
        f.guess_matrix(im)
        f.fit_matrix(im)
        print f.result
        #plt.imshow(im)
        #plt.show()


if __name__ == "__main__":
    get_errorbars('20170407-0055beam_mon', axis=0)
