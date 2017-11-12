import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import pylab as pl
from plot_tools import *
import matplotlib as mpl
import brewer2mpl as cb
from matplotlib import rcParams
import brewer2mpl as cb
import cmocean as cm
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.lines as mlines

if __name__ == "__main__":
    oos_data = pd.read_csv('out_of_sample_data_with_p20_s4.csv')
    xdiffs = oos_data.diff()[1:][['x_t', 'x_l']]
    print xdiffs.corr()

    mean_xt, mean_xl = np.mean(oos_data.x_t), np.mean(oos_data.x_l)
    oos_data.x_t -= mean_xt
    oos_data.x_ct -= mean_xt
    oos_data.x_l -= mean_xl
    oos_data.x_cl -= mean_xl

    scl = 1/1.61
    fig_width = 3
    fig_height = scl*fig_width

    rcpars.update({
        "figure.autolayout": True,
        "figure.figsize": (fig_width, fig_height),
        "figure.subplot.left": .1,
        "figure.subplot.right": .94,
        "figure.subplot.bottom": .05,
        "figure.subplot.top": .95,
    })

    pl.rcParams.update(rcpars)

    labels = ['a', 'b', 'c']
    init_plt()
    fig = pl.figure()
    ax0 = fig.add_subplot(1, 1, 1)
    slope, intercept, r_value, p_value, std_err = stats.linregress(oos_data.x_ct, oos_data.x_t)
    print slope
    print intercept

    print r_value
    print p_value

    xlin = np.linspace(oos_data.min().x_t, oos_data.max().x_t, 10)

    ax0.plot(oos_data.x_ct, oos_data.x_t, linewidth=0, marker = 'o')
    ax0.plot(xlin, slope*xlin + intercept, linewidth=0, marker = 'o')
    plt.show()

