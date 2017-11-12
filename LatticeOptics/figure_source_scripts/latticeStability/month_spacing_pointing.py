import numpy as np
import pyfits
import matplotlib.pyplot as plt
import pylab as pl
from plot_tools import *
import matplotlib as mpl
import brewer2mpl as cb
from matplotlib import rcParams
import brewer2mpl as cb
import cmocean as cm
import matplotlib.gridspec as gridspec
import pickle as pkl
import os
import re
import pandas as pd

def find_matching_dirs(base_dir, regex=".*", conditions=[]):
    """
    given a directory, returns paths to all subdirectories whose name satisfies the regex
    :param base_dir:
    :param regex:
    :param condition: list of lambda functions used in a filter that must all be satisfied
    :return:
    """
    sub_dir_names = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]
    sub_dir_nums = filter(lambda x: re.match(regex, x), sub_dir_names)
    for condition in conditions:
        sub_dir_nums = filter(condition, sub_dir_nums)
    num_dirs = map(lambda x: os.path.join(base_dir, x), sub_dir_nums)
    return zip(num_dirs, sub_dir_nums)

def sample_cmap(cmap, n_pts, bottom=0.0, top=1.0, gamma=1):
    """
    Produces n_pts colors sampled from the given cmap
    :param cmap: colormap to be sampled from. Assume it is a function on [0, 1] that maps to a color tuple
    :param n_pts: number of points to sample
    :param bottom: minimum input
    :param top: maximum input
    :param gamma: corresponds to nonlinearity
    :return: list of color tuples sampled from cmap
    """
    assert top <= 1
    assert bottom >= 0
    assert top >= bottom
    rng = float(top) - float(bottom)
    return map(cmap, bottom + rng*np.power(np.linspace(0, 1, n_pts), 1))

def extract_stabilities(runlog_dir="W:\\RunLog"):
    dated_lvecs = []
    years = find_matching_dirs(runlog_dir, regex="[0-9]{4}")#, conditions=[lambda x: int(x) in [2016]])
    for year, n_year in years:
        months = find_matching_dirs(year, regex="[0-9]{2}")#, conditions=[lambda x: int(x) in [6]])
        for month, n_month in months:
            days = find_matching_dirs(month, regex="[0-9]{2}")
            for day, n_day in days:
                print "analyzing {year}-{month}-{day}".format(year=n_year, month=n_month, day=n_day)
                scans = find_matching_dirs(day, regex="[0-9]{8}-[0-9]{4}")
                current_max = 0
                lvecs = None
                best_scan = None
                for scan, n_scan in scans:
                    if find_matching_dirs(scan, regex="FitResults"):
                        try:
                            fits = filter(lambda x: re.match('fitresult_[0-9]{3}-[0-9]{2}.pkl', x), os.listdir(os.path.join(scan, "FitResults")))
                            nfits = len(fits)
                            if nfits > current_max:
                                with open(os.path.join(scan, "FitResults", "fitresult_000-00.pkl"), mode='rb') as f:
                                    p = pkl.load(f)
                                    lvecs = p['lvec']
                                    best_scan = n_scan
                        except:
                            print "lvecs for {} failed to save".format(n_scan)
                            #print lvecs
                if lvecs is not None:
                    dated_lvec = (n_year, n_month, n_day, best_scan, nfits, lvecs)
                    print "saving lvecs for {}".format(n_scan)
                    #print dated_lvec
                    dated_lvecs.append(dated_lvec)
    pkl.dump(dated_lvecs, open('dated_lvecs.pkl', mode='wb'))


def load_dated_lvecs(filename):
    dated_lvecs = pkl.load(open(filename, mode='rb'))
    return dated_lvecs


if __name__ == "__main__":
    raw_data = load_dated_lvecs('dated_lvecs.pkl')
    years = []
    months = []
    days = []
    lv_mat = None
    for entry in raw_data:
        year, month, day, scan, num, lvecs = entry
        years.append(year)
        months.append(month)
        days.append(day)
        lv_mat = lvecs if lv_mat is None else np.vstack((lv_mat, lvecs))
    print lv_mat
    date_frame = pd.DataFrame({'year': years, 'month': months, 'day': days})
    date_frame = pd.to_datetime(date_frame)

    lv_frame = pd.DataFrame(data=lv_mat, columns=['lv0', 'lv1', 'offset0', 'angle0', 'angle1', 'offset1'])
    stability_frame = pd.concat([date_frame, lv_frame], axis=1)
    outlier_dates = [pd.to_datetime('08/08/2016')]
    print outlier_dates

    stability_frame = stability_frame.set_index(0)
    stability_frame.index.names = ['Date']
    #stability_frame = stability_frame.rename(index={0: "date"})

    stability_frame = stability_frame.drop(outlier_dates)
    #print stability_frame


    # PLOTTING
    fig_name = "lattice_stability"
    rcParams.update({})
    scl = 1.61
    fig_width = 4
    fig_height = scl*fig_width

    rcpars.update({
        "figure.autolayout": True,
        "figure.figsize": (fig_width, fig_height),
        "figure.subplot.left": .25,
        "figure.subplot.right": .94,
        "figure.subplot.bottom": .2,
        "figure.subplot.top": .95,
    })

    pl.rcParams.update(rcpars)

    init_plt()
    fig = pl.figure()
    ax0 = fig.add_subplot(4, 1, 1)
    ax1 = fig.add_subplot(4, 1, 2)
    ax2 = fig.add_subplot(4, 1, 3)
    ax3 = fig.add_subplot(4, 1, 4)
    subplots = [ax0, ax1, ax2, ax3]
    cols = ['lv0', 'lv1', 'angle0', 'angle1']

    #subplots = stability_frame.loc[:, ['lv0', 'lv1', 'angle0', 'angle1']].plot(subplots=True, legend=False, marker='o',
                                                                               #ms=4, ax=ax, linewidth=1,
                                                                               #colormap=cb.get_map('Set1', 'Qualitative', 4).mpl_colormap)
    colors = sample_cmap(cm.cm.haline, 5, gamma=1, top=0.5)[0:2] + list(reversed(cb.get_map('OrRd', 'Sequential', 6, reverse=False).mpl_colors[4:]))

    for s, c, color in zip(subplots, cols, colors):
        stability_frame[c].plot(legend=False, marker='o', linewidth=1, ms=4, ax=s, color=color)
    print stability_frame['01-01-2017':].std()
    print stability_frame['01-01-2017':].mean()

    spring = stability_frame['01-01-2017':]
    df = spring['angle0']-spring['angle1']
    print df.std()
    print cb.get_map('Set1', 'Qualitative', 4).mpl_colormap

    for l in ['lv0', 'lv1']:
        print stability_frame['01-01-2017':].std()[l]/stability_frame['01-01-2017':].mean()[l]
    ylims=[(7.4, 9.1), (7.4, 9.1), (47.1, 47.8), (-41.5, -40.8)]
    ylabels = (r'$|v_x|$ (pix)', r'$|v_y|$ (pix)', r'$\theta_x$ (deg)', r'$\theta_y$ (deg)')
    labeled_plot_idxs = [3]
    for idx, (subplot, ylim, ylabel) in enumerate(zip(subplots, ylims, ylabels)):
        subplot.set_ylim(ylim)
        subplot.set_ylabel(ylabel)
        if idx not in labeled_plot_idxs:
            subplot.set_xticks([])
            subplot.set_xlabel('')
        #subplot.tick_params(rotation=70)

    plt.setp(plt.xticks()[1], rotation=45)
    plt.savefig('{}.pdf'.format(fig_name))
    plt.show()

