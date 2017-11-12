import numpy as np
from AnalyzeStability import *


def analyze_stability_scans():

    date = (2016, 9, 21)
    #scan_list = [52, 57, 59, 61, 62, 63, 64]
    scan_list = [9]

    for scan in scan_list:
        print "analyzing scan %s" % scan
        measurement = Measurement(date, scan, base_directory='W:RunLog')
        measurement.plot()
        plt.show()


if __name__ == "__main__":
    analyze_stability_scans()