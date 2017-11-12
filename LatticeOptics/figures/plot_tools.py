import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import brewer2mpl
from matplotlib.ticker import MultipleLocator
import matplotlib.colors as colors
from plot_settings import *
import matplotlib.ticker as mtick
mpl.use('pdf')

'''
plt     plot to output to
th_x    array of theory x points (plotted as line)
th_y    theory y
data_x  data x (plotted as points)
data_y  data y
err_x   data error x
err_y   data error y
cs      start color, i.e. color of first set of points (-1)
cf      final color
cpow    exponent to use for nonlinear coloring

Optional:
xlim    x axis limits
ylim    y axis limits
ymax    maximum value for y, will truncate error bars past this
label   label to place in upper right corner
tick_format     format of the numbers for each tick
trunc_marg      margin to truncate points for (in x direction)
                must set xlim!
'''

# Global settings
mew = 0.8  # Marker Edge Width #0.8 #
ms = 1.2  # Marker Size #1.2 #
cmap = brewer2mpl.get_map('RdBu', 'Diverging', 11).mpl_colormap


def init_plt():
    plt.rc('font', family='sans-serif')
    plt.rcParams['font.sans-serif'] = 'Helvetica'
    plt.rcParams['font.size'] = 7
    plt.rcParams['mathtext.fontset'] = 'custom'

    plt.rcParams['mathtext.rm'] = 'Helvetica'
    plt.rcParams['mathtext.it'] = 'Helvetica:italic'
    plt.rcParams['mathtext.bf'] = 'Helvetica:bold'

    plt.rcParams['legend.frameon'] = False

    plt.rcParams['axes.linewidth'] = 0.4
    plt.rcParams['xtick.major.width'] = 0.3
    plt.rcParams['xtick.major.size'] = 1
    plt.rcParams['ytick.major.width'] = 0.3
    plt.rcParams['ytick.major.size'] = 1

    plt.rcParams['xtick.minor.width'] = 0.2
    plt.rcParams['xtick.minor.size'] = 0.8
    plt.rcParams['ytick.minor.width'] = 0.2
    plt.rcParams['ytick.minor.size'] = 0.8

    plt.rcParams.update({'font.sans-serif': 'Helvetica',
                         'font.weight': '200',
                         'pdf.fonttype': 42,
                         'axes.labelsize': 5,
                         'xtick.labelsize': 5,
                         'ytick.labelsize': 5,
                         'legend.fontsize': 5,
                         'lines.linewidth': 0.62})


def plot_fft(ax, fft, vmax=None, labels=(True, True), shortlbl=False):
    if vmax is None:
        vmax = 0.2
    # cmap = brewer2mpl.get_map('PuBu', 'Sequential', 9).mpl_colormap
    cm = brewer2mpl.get_map('BrBG', 'Diverging', 9).mpl_colormap
    ax.pcolormesh(fft, vmin=-vmax, vmax=vmax, cmap=cm, rasterized=True)

    # ============plot labels===============
    if labels[0]:
        if shortlbl:
            ax.set_xlabel(r"$p/ \,\hbar$")
        else:
            ax.set_xlabel(r"$\mathrm{lattice \, momentum} \, p\mathrm{/} \,\hbar$")
    if labels[1]:
        if shortlbl:
            ax.set_ylabel(r"$p/ \,\hbar")
        else:
            ax.set_ylabel(r"$\mathrm{lattice \, momentum} \, p\mathrm{/} \,\hbar$")

    if shortlbl:
        ax.xaxis.set_label_coords(0.5, -0.2)
    else:
        ax.xaxis.set_label_coords(0.5, -0.15)
    ax.yaxis.set_label_coords(-0.23, 0.5)

    # ============axes limits===============
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    # ============ticks===============
    plt.sca(ax)
    plt.xticks([0, 5.5], [0, r'$\pi/a$'])
    plt.yticks([0, 5.5], [0, r'$\frac{\pi}{a}$'])
    if not labels[0]:
        ax.set_xticklabels([])
    if not labels[1]:
        ax.set_yticklabels([])

    ax.tick_params(pad=2)

    return cm


def plot_fft_symm(ax, fft, vmax=None, labels=(True, True), shortlbl=False):
    # Wrap 0 value over to 2pi/a
    fft[10, :] = fft[0, :]
    fft[:, 10] = fft[:, 0]

    if vmax is None:
        vmax = 0.2
    # cmap = brewer2mpl.get_map('PuBu', 'Sequential', 9).mpl_colormap
    cm = brewer2mpl.get_map('BrBG', 'Diverging', 9).mpl_colormap
    ax.pcolormesh(fft, vmin=-vmax, vmax=vmax, cmap=cm, rasterized=True)

    # ============plot labels===============
    if labels[0]:
        if shortlbl:
            # ax.set_xlabel(r"$p/ \,\hbar")
            ax.set_xlabel(r"$q_x \, (1\!/\!a)$")
        else:
            # ax.set_xlabel(r"$\mathrm{lattice \, momentum} \, p\mathrm{/} \,\hbar$")
            ax.set_xlabel(r"$\mathrm{momentum} \, q_x \, (1\!/\!a)$")
    if labels[1]:
        if shortlbl:
            # ax.set_ylabel(r"$p/ \,\hbar")
            ax.set_ylabel(r"$q_y \, (1\!/\!a)$")
        else:
            # ax.set_ylabel(r"$\mathrm{lattice \, momentum} \, p\mathrm{/} \,\hbar$")
            ax.set_ylabel(r"$\mathrm{momentum} \, q_y \, (1\!/\!a)$")

    if shortlbl:
        ax.xaxis.set_label_coords(0.5, -0.3)
    else:
        ax.xaxis.set_label_coords(0.5, -0.2)
    ax.yaxis.set_label_coords(-0.23, 0.5)

    # ============axes limits===============
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)

    # ============ticks===============
    plt.sca(ax)

    # plt.xticks([0.5, 5.5, 10.5], [0, r'$\pi\!/\!a$', r'$\frac{2\pi}{a}$'])
    # plt.yticks([0.5, 5.5, 10.5], [0, r'$\frac{\pi}{a}$', r'$\frac{2\pi}{a}$'])

    plt.xticks([0.5, 5.5, 10.5], ["", r'$\pi$', ""])#["", r'$\frac{\pi}{a}$', ""])
    ax.set_xticks([0.8, 9.8], minor=True)
    ax.set_xticklabels([0, r'$2\pi$'], minor=True)
    # ax.set_xticklabels([0, r'$\frac{2\pi}{a}$'], minor=True)
    plt.yticks([0.5, 5.5, 10.5], [0, r'$\pi$', r'$2\pi$'])
    # plt.yticks([0.5, 5.5, 10.5], [0, r'$\frac{\pi}{a}$', r'$\frac{2\pi}{a}$'])

    for line in ax.xaxis.get_minorticklines():
        line.set_visible(False)

    if not labels[0]:
        ax.set_xticklabels([])
    if not labels[1]:
        ax.set_yticklabels([])

    ax.tick_params(pad=2.0)
    ax.tick_params(axis='y', pad=1.5)
    ax.tick_params(which='minor', pad=2)

    return cm


class PowerNormSymm(colors.Normalize):
    def __init__(self, gamma=1.0, vmin=None, vmax=None, clip=False):
        self.gamma = gamma
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        return np.ma.masked_array(0.5 + np.sign(value)*np.power(np.abs(value), self.gamma) /
                                  (2*np.power(self.vmax, self.gamma)))


def plot_corrmap(ax, corrmap, vmax=None, labels=(True, True)):
    #ax.fill([38, 60, 60, 38], [38, 38, 60, 60], fill=False, hatch=8*'//', zorder=1, edgecolor=color_grey)

    corrmap = np.where(corrmap == -2, np.nan, corrmap)
    # cmapRdBu = brewer2mpl.get_map('RdBu', 'Diverging', 7).mpl_colormap
    # cmapRdBu = brewer2mpl.get_map('Greys', 'Sequential', 7).mpl_colormap
    # cmapRdBu = mpl.cm.get_cmap('GnBu_r')

    import cubehelix
    # cmapTop = cubehelix.cmap(start=1.25, rot=0., minLight=0.6, sat=1)(np.linspace(0, 1, 128))
    # cmapBot = cubehelix.cmap(start=-0.25, rot=0., maxLight=0.6, sat=1)(np.linspace(0, 1, 128))
    # combined = np.vstack((cmapBot, cmapTop))
    # cmapRdBu = mpl.colors.LinearSegmentedColormap.from_list('cmapRdBu', combined)
    # cmapRdBu = cubehelix.cmap(start=-0.3, rot=0, sat=0.5)
    cmapRdBu = cubehelix.cmap(startHue=330, rot=-0.05, sat=0.7, minSat=0.5, maxSat=0.5)

    cmapRdBu.set_bad(color=color_grey_ll)
    if vmax is None:
        vmax = 0.2
        vmin = -vmax
    else:
        vmin = -vmax
    # norm = PowerNormSymm(gamma=0.4, vmin=vmin, vmax=vmax, clip=False)
    norm = PowerNormSymm(gamma=0.7, vmin=vmin, vmax=vmax, clip=False)
    # norm = PowerNormSymm(gamma=1, vmin=vmin, vmax=vmax, clip=False)
    ax.pcolormesh(corrmap, norm=norm, cmap=cmapRdBu, zorder=2, rasterized=True)

    # ============plot labels===============
    if labels[0]:
        ax.set_xlabel(r"$d_x \, \mathrm{(sites)}$")
    if labels[1]:
        ax.set_ylabel(r"$d_y \, \mathrm{(sites)}$")

    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.yaxis.set_label_coords(-0.23, 0.5)

    # ============axes limits===============
    ax.set_xlim(39, 60)
    ax.set_ylim(39, 60)

    # ============ticks===============
    plt.sca(ax)
    plt.xticks([39.5, 49.5, 59.5], [-10, 0, 10])
    plt.yticks([39.5, 49.5, 59.5], [-10, 0, 10])
    if not labels[0]:
        ax.set_xticklabels([])
    if not labels[1]:
        ax.set_yticklabels([])

    ax.tick_params(pad=2)

    return norm, vmin, vmax, cmapRdBu


def plot_corrfn(ax, dist, corr, err, labels=(True, True)):
    ax.errorbar(dist, corr, yerr=err,
                fmt="o", markeredgecolor=color_blue_d, markerfacecolor=color_blue_l, capsize=0, ecolor=color_blue_d,
                markeredgewidth=0.25, elinewidth=0.5, markersize=3.0, label=None, zorder=1)

    # ax.errorbar([8, 12], [0.36**2, 0.36**2],
    #             color='grey', label=None, zorder=2, linewidth=0.3, dashes=[1, 1])

    hline_settings = dict(linestyle='-', color='grey', zorder=0, linewidth=0.3, dashes=[1, 1])
    ax.axhline(y=0, **hline_settings)

    #============plot labels===============
    if labels[0]:
        ax.set_xlabel(r"$d \, \mathrm{(sites)}$")
    else:
        ax.set_xticklabels([])

    if labels[1]:
        ax.set_ylabel(r"$(-1)^i \, C_{d}$")
        # ax.annotate("Heisenberg", xy=(4.7, 0.36**2+0.04), xycoords='data', fontsize=5, color=color_grey_d)
        # ax.annotate(r"$T=0$", xy=(6.8, 0.36**2+0.01), xycoords='data', fontsize=5, color=color_grey_d)
    else:
        ax.set_yticklabels([])

    ax.xaxis.set_label_coords(0.5, -0.1)
    ax.yaxis.set_label_coords(-0.23, 0.5)

    #============axes limits===============
    ax.set_xlim(0.5, 9.5)
    ax.set_ylim(-0.02, 0.305)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.tick_params(pad=2)

    return


def plot_corrfn_fit(ax, fit, isfit=False):
    # # fit = FitFunctionLambda(lambda x, a, xi: a*np.exp(-x/xi), [0.2, 5.0])
    # fit = FitFunctionLambda(lambda x, xi: 0.0576*np.exp(-x/xi), [5.0])
    # fit.fit(dist[3:], corr[3:], sigma=err[3:], absolute_sigma=True)
    # print fit.result
    distfit, corrfit = fit.generate_fitted_data()

    if isfit:
        color = color_blue_m
    else:
        color = "grey"

    ax.errorbar(distfit, corrfit,
                fmt='-', color=color, linewidth=0.5, label=None, zorder=2)
    return fit.result['xi']


def plot_corrfn_combinedfit(axes, fit, select, x):

    a = fit.result['a0'].bare_value
    xi = np.empty(len(select))
    xi_err = np.empty(len(select))
    ax_i = [sum(select[:i])*select[i] for i in range(len(select))]
    for key in fit.result:
        if key == 'a0':
            pass
        elif not select[int(key[-1])-1]:
            xi[int(key[-1])-1] = fit.result[key].bare_value
            xi_err[int(key[-1])-1] = fit.result[key].bare_error
        else:
            y = a*np.exp(-x/fit.result[key].bare_value)
            axes[ax_i[int(key[-1])-1]].errorbar(x, y,
                                          fmt='-', color=color_blue_m, linewidth=0.5, label=None, zorder=2)
            xi[int(key[-1])-1] = fit.result[key].bare_value
            xi_err[int(key[-1])-1] = max(fit.result[key].bare_error, 1e-8)  # This is a bit of a hack...
        if key == 'a1':
            for ax in axes[1:]:
                ax.errorbar(x, y,
                            fmt='-', color='grey', linewidth=0.5, label=None, zorder=2)

    return xi, xi_err


def get_corrmap_stagmag(d_corr_map, d_corr_map_err, mults):
    corrmag = []
    corrmag_err = []

    for i in range(len(d_corr_map)):
        dcorr = d_corr_map[i]['data']
        dcorr_err = d_corr_map_err[i]['data']
        h, w = dcorr.shape
        checkerboard = np.row_stack((h+1)/2*(np.r_[(w+1)/2*[1, -1]], np.r_[(w+1)/2*[-1, 1]]))[1:, 1:]
        dmult = mults[i]
        center = np.unravel_index(np.argmax(dmult), dmult.shape)
        dmult[center] = 0
        dcorr = np.where(dcorr == -2, 0, dcorr)
        corrmag += [np.sqrt(np.sum(dcorr * dmult * checkerboard)/np.sum(dmult))]
        corrmag_err += [np.sqrt(np.sum(dmult**2 * dcorr_err**2 / (4 * np.sum(dcorr * dmult * checkerboard) *
                                                                  np.sum(dmult))))]

    return corrmag, corrmag_err


def get_corrmap_stagmag_nonn(d_corr_map, d_corr_map_err, mults):
    corrmag = []
    corrmag_err = []

    for i in range(len(d_corr_map)):
        dcorr = d_corr_map[i]['data']
        dcorr_err = d_corr_map_err[i]['data']
        h, w = dcorr.shape
        checkerboard = np.row_stack((h+1)/2*(np.r_[(w+1)/2*[1, -1]], np.r_[(w+1)/2*[-1, 1]]))[1:, 1:]
        dmult = mults[i]
        center = np.unravel_index(np.argmax(dmult), dmult.shape)
        dmult[center] = 0
        dmult[center[0]-1, center[1]  ] = 0
        dmult[center[0]  , center[1]-1] = 0
        dmult[center[0]+1, center[1]  ] = 0
        dmult[center[0]  , center[1]+1] = 0
        # d = 2
        # dmult[center[0]-d:center[0]+d+1, center[1]-d:center[1]+d+1] = 0
        dcorr = np.where(dcorr == -2, 0, dcorr)
        corrmag += [np.sqrt(np.sum(dcorr * dmult * checkerboard)/np.sum(dmult))]
        corrmag_err += [np.sqrt(np.sum(dmult**2 * dcorr_err**2 / (4 * np.sum(dcorr * dmult * checkerboard) *
                                                                  np.sum(dmult))))]

    return corrmag, corrmag_err


def plot_magscan_comp(ax, stagmag, stagmag_err, corrmag, corrmag_err, color='blue'):
    if color == 'blue':
        mec = color_blue_d
        mfc = color_blue_l
        label = "Temperature"
    elif color == 'red':
        mec = color_red_d
        mfc = color_red_l
        label = "Density"

    ax.errorbar(stagmag, corrmag, xerr=stagmag_err, yerr=corrmag_err,
                fmt="o", markeredgecolor=mec, markerfacecolor=mfc, capsize=0, ecolor=mec,
                markeredgewidth=0.25, elinewidth=0.5, markersize=3.5, label=label, zorder=2)

    #============plot labels===============
    ax.set_xlabel(r"$m^z_c$" + "  (from staggered magnetization)")
    ax.set_ylabel(r"$m^z_c$" + "  (from correlation map)")

    ax.xaxis.set_label_coords(0.5, -0.05)
    ax.yaxis.set_label_coords(-0.06, 0.5)

    #============axes limits===============
    ax.set_xlim(-0.02, 0.32)
    ax.set_ylim(-0.02, 0.32)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))

    ax.tick_params(pad=2)

    return


def plot_magscan(ax, x, x_err, mag, mag_err, x2=None, include0=False, markersize=3.5, color=None):
    if color == 'blue':
        mec = color_blue_d
        mfc = color_blue_l
        ecolor = color_blue_d
        label = r'$m^z_c$'
    elif color == 'red':
        mec = color_red_d
        mfc = color_red_l
        ecolor = color_red_d
        label = 'From correlation function'
    else:
        mec = color_blue_d
        mfc = color_blue_l
        ecolor = color_blue_d
        label = 'Experimental data'

    ax.errorbar(x, mag, xerr=x_err, yerr=mag_err,
                fmt="o", markeredgecolor=mec, markerfacecolor=mfc, capsize=0, ecolor=ecolor,
                markeredgewidth=0.25, elinewidth=0.5, markersize=markersize, label=label, zorder=3)

    if include0:
        ax.errorbar([0], [0.36], yerr=[0],
                    fmt="*", markeredgecolor=color_green_d, markerfacecolor=color_green_m, capsize=0, ecolor=color_green_d,
                    markeredgewidth=0.25, elinewidth=0.5, markersize=4.5, label=None, zorder=2)

    hline_settings = dict(linestyle='-', color='grey', zorder=0,linewidth=0.3, dashes=[1, 1])
    ax.axhline(y=0.0, **hline_settings)

    #============plot labels===============
    ax.set_xlabel(r"$T/t$")
    ax.set_ylabel(r"$m^z_c$")

    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    #============axes limits===============
    # ax.set_xlim(-0.05, 1.35)
    ax.set_xlim(0.18, 1.35)
    # if include0:
    ax.set_ylim(-0.01, 0.37)
    # else:
    #     ax.set_ylim(-0.01, 0.17)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))

    ax.tick_params(pad=2)
    #============second x-axis===============
    if x2 is not None:
        ax2 = ax.twiny()

        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.tick_top()
        ax2.set_xticks([x[i] for i in [0, 3, 4, 5]])
        ax2.set_xticklabels([x2[i] for i in [0, 3, 4, 5]])
        ax2.set_xlabel(r"$\mathrm{Hold \, time \,(} s \mathrm{)}$")
        ax2.xaxis.set_label_coords(0.5, 1.1)

        ax2.tick_params(pad=2)
    else:
        ax2 = None

    return ax2


# From Subir
def get_stagmag_theory(Ts, N_0, C_xi, rho_s, mult):
    sums = []
    try:
        _ = (e for e in Ts)
    except TypeError:
        Ts = [Ts]

    for T in Ts:
        xi_T = C_xi * np.exp(2*np.pi*rho_s/T)
        where = np.argwhere(mult)
        (ystart, xstart), (ystop, xstop) = where.min(0), where.max(0) + 1
        m = np.copy(mult[ystart:ystop, xstart:xstop])
        center = np.unravel_index(np.argmax(m), m.shape)

        summand = np.zeros_like(m)

        for j in range(m.shape[0]):
            for k in range(m.shape[1]):
                if np.abs(j-center[0]) < 1 and np.abs(k-center[1]) < 1:
                    summand[j, k] = 0
                    m[j, k] = 0
                else:
                    summand[j, k] = np.exp(-np.sqrt((j-center[0])**2 + (k-center[1])**2)/xi_T)

        summand = np.multiply(summand, m)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # cax = plt.imshow(summand, interpolation='nearest')
        # plt.colorbar(cax)
        # plt.show()
        # sums += [N_0 * np.sum(summand) / np.sum(m)]
        sums += [np.sqrt(N_0 * np.sum(summand) / np.sum(m))]

    return sums


def get_stagmag_theory_adj(Ts, N_0, C_xi, rho_s, mult):
    sums = []
    try:
        _ = (e for e in Ts)
    except TypeError:
        Ts = [Ts]

    for T in Ts:
        xi_T = C_xi * np.exp(2*np.pi*rho_s/T)
        where = np.argwhere(mult)
        (ystart, xstart), (ystop, xstop) = where.min(0), where.max(0) + 1
        m = np.copy(mult[ystart:ystop, xstart:xstop])
        center = np.unravel_index(np.argmax(m), m.shape)

        summand = np.zeros_like(m)

        for j in range(m.shape[0]):
            for k in range(m.shape[1]):
                if np.abs(j-center[0]) < 1 and np.abs(k-center[1]) < 1:
                    summand[j, k] = 0
                    m[j, k] = 0
                if np.sqrt((j-center[0])**2 + (k-center[1])**2) < xi_T:
                    summand[j, k] = np.exp(-np.sqrt((j-center[0])**2 + (k-center[1])**2)/xi_T)
                else:
                    summand[j, k] = (np.sqrt((j-center[0])**2 + (k-center[1])**2)/xi_T)**(-1/2.) * \
                                    np.exp(-np.sqrt((j-center[0])**2 + (k-center[1])**2)/xi_T) * \
                                    (1 - xi_T/np.sqrt((j-center[0])**2 + (k-center[1])**2))

        summand = np.multiply(summand, m)
        # import matplotlib.pyplot as plt
        # plt.figure()
        # cax = plt.imshow(summand, interpolation='nearest')
        # plt.colorbar(cax)
        # plt.show()
        # sums += [N_0 * np.sum(summand) / np.sum(m)]
        sums += [np.sqrt(N_0 * np.sum(summand) / np.sum(m))]

    return sums


def plot_magscan_theory(ax, f, T, N, C, rho, mult):

    ax.errorbar(T, f(T, N.bare_value, C.bare_value, rho.bare_value, mult),
                fmt='-', color=color_set_r, linewidth=0.5, label='Fitted model', zorder=1)

    # vals = []
    # N = [N.bare_value - N.bare_error, N.bare_value + N.bare_error]
    # C = [C.bare_value - C.bare_error, C.bare_value + C.bare_error]
    # rho = [rho.bare_value - rho.bare_error, rho.bare_value + rho.bare_error]
    #
    # for n in N:
    #     for c in C:
    #         for r in rho:
    #             vals += [f(T, n, c, r, mult)]
    #
    # vals_u = np.max(vals, 0)
    # vals_l = np.min(vals, 0)
    #
    # plt.sca(ax)
    # plt.fill_between(np.linspace(0, 1.4, 100), vals_l, vals_u, facecolor=color_set_r_l, alpha=0.3, edgecolor='none',
    #                  zorder=1)

    return


def plot_magscan_QMC(ax, temp, stagmag, stagmag_err):

    ax.errorbar(temp, stagmag, yerr=stagmag_err,
                fmt="-o", markeredgecolor=color_grey_d, markerfacecolor=color_grey_l, capsize=0, ecolor=color_grey_d,
                markeredgewidth=0.25, elinewidth=0.25, markersize=1.5, label='QMC theory', zorder=2, linewidth=0.5, color=color_grey_d)

    # ax.errorbar(temp, stagmag,
    #             fmt="-", color=color_grey, linewidth=0.5, label='0 parameters', zorder=1)
    #
    # vals_l = stagmag-stagmag_err
    # vals_u = stagmag+stagmag_err
    #
    # plt.sca(ax)
    # plt.fill_between(temp, vals_l, vals_u, facecolor=color_grey_l, alpha=0.3, edgecolor='none',
    #                  zorder=1)

    return


def plot_xiscan(ax, x, x_err, xi, xi_err, x2=None):
    ax.errorbar(x, xi, xerr=x_err, yerr=xi_err,
                fmt="o", markeredgecolor=color_blue_d, markerfacecolor=color_blue_l, capsize=0, ecolor=color_blue_d,
                markeredgewidth=0.25, elinewidth=0.5, markersize=3.5, label=None, zorder=2)

    hline_settings = dict(linestyle='-', color='grey', zorder=0,linewidth=0.3, dashes=[1, 1])
    ax.axhline(y=0.0, **hline_settings)

    #============plot labels===============
    ax.set_xlabel(r"$T/t$")
    ax.set_ylabel(r"$\xi \, \mathrm{(sites)}$")

    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    #============axes limits===============
    # ax.set_xlim(-0.05, 1.35)
    ax.set_xlim(0.18, 1.35)
    ax.set_ylim(-0.5, 10.5)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(2))
    ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.tick_params(pad=2)

    #============second x-axis===============
    if x2 is not None:
        ax2 = ax.twiny()

        ax2.set_xlim(ax.get_xlim())
        ax2.xaxis.tick_top()
        ax2.set_xticks([x[i] for i in [0, 3, 4, 5]])
        ax2.set_xticklabels([x2[i] for i in [0, 3, 4, 5]])
        ax2.set_xlabel(r"$\mathrm{Hold \, time \,(} s \mathrm{)}$")
        ax2.xaxis.set_label_coords(0.5, 1.1)

        ax2.tick_params(pad=2)
    else:
        ax2 = None

    return ax2


def plot_xiscan_inset(ax, x, xi, xi_err, fit):
    ax.errorbar(1/np.array(x), xi, yerr=xi_err,
                fmt="o", markeredgecolor=color_blue_d, markerfacecolor=color_blue_l, capsize=0, ecolor=color_blue_d,
                markeredgewidth=0.25, elinewidth=0.5, markersize=3.5, label=None, zorder=2)

    xfit, xifit = fit.generate_fitted_data(np.linspace(0.15, max(x)*3.0, 100))

    ax.errorbar(1/np.array(xfit), xifit,
                fmt='-', color=color_blue_m, linewidth=0.5, label=None, zorder=2)

    # hline_settings = dict(linestyle='-', color='grey', zorder=0,linewidth=0.3, dashes=[1, 1])
    # ax.axhline(y=0.0, **hline_settings)

    #============plot labels===============
    ax.set_xlabel(r"$(T\!/{}t)^{-1}$")
    # ax.set_xlabel(r"$t/T$")
    ax.set_ylabel(r"$\xi \, \mathrm{(sites)}$")

    ax.xaxis.set_label_coords(0.5, -0.15)
    ax.yaxis.set_label_coords(-0.2, 0.5)

    #============axes limits===============
    ax.set_yscale("log", nonposx='clip')

    ax.set_xlim(0.5, 5.5)
    ax.set_ylim(0.2, 40)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.2))
    ax.yaxis.set_major_locator(mpl.ticker.LogLocator(base=10.0))
    #ax.yaxis.set_minor_locator(mpl.ticker.LogLocator(base=1.0))

    ax.tick_params(pad=2)

    return


def plot_magscan_density(ax, x, x_err, mag, mag_err, x2=None, markersize=3.5, color=None):
    if color == 'blue':
        mec = color_blue_d
        mfc = color_blue_l
        ecolor = color_blue_d
        label = r'$m^z_c$'
    elif color == 'red':
        mec = color_red_d
        mfc = color_red_l
        ecolor = color_red_d
        label = r'$d < 2$ contributions omitted'
    else:
        mec = color_blue_d
        mfc = color_blue_l
        ecolor = color_blue_d
        label = None

    ax.errorbar(x, mag, yerr=mag_err, xerr=x_err,
                fmt="o", markeredgecolor=mec, markerfacecolor=mfc, capsize=0, ecolor=ecolor,
                markeredgewidth=0.5, markersize=markersize, label=label, zorder=2)

    hline_settings = dict(linestyle='-', color='grey', zorder=0, linewidth=0.3, dashes=[1, 1])
    ax.axhline(y=0.0, **hline_settings)

    #============plot labels===============
    ax.set_xlabel(r"$\mathrm{Singles \, density} \, n_s$")
    ax.set_ylabel(r"$m^z_c$")

    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.15, 0.5)

    #============axes limits===============
    ax.set_xlim(1.0, 0.72)
    ax.set_ylim(-0.01, 0.32)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))

    ax.tick_params(pad=2)

    # ============second x-axis===============
    if x2 is not None:
        ax2 = ax.twiny()
        ax2.get_xaxis().tick_bottom()
        ax.get_xaxis().tick_top()
        ticks_bot = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25']
        ticks_top = [np.max(x)-float(tick)/x2 for tick in ticks_bot]

        # ============new labels===============
        ax.xaxis.set_label_coords(0.5, 1.13)
        ax2.set_xlabel(r"$\mathrm{Doping} \, \delta$")
        ax2.xaxis.set_label_coords(0.5, -0.12)

        # ============new axes limits===============
        ax2.set_xlim(ticks_top[0]+0.008, ticks_top[-1]-0.008)
        ax.set_xlim(ax2.get_xlim())

        # ============new ticks===============
        ax2.set_xticks(ticks_top)
        ax2.set_xticklabels(ticks_bot)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        # ax2.tick_params('x', bottom='off', length=0)

        ax2.tick_params(pad=2)
    else:
        ax2 = None

    return ax2


def plot_xiscan_density(ax, x, x_err, xi, xi_err, x2=None, color='blue'):
    if color == 'blue':
        mec = color_blue_d
        mfc = color_blue_l
        ecolor = color_blue_d
    elif color == 'grey':
        mec = color_grey_d
        mfc = color_grey_l
        ecolor = color_grey_d

    ax.errorbar(x, xi, xerr=x_err, yerr=xi_err,
                fmt="o", markeredgecolor=mec, markerfacecolor=mfc, capsize=0, ecolor=ecolor,
                markeredgewidth=0.5, elinewidth=0.5, markersize=3.5, label=None, zorder=2)

    hline_settings = dict(linestyle='-', color='grey', zorder=0, linewidth=0.3, dashes=[1, 1])
    ax.axhline(y=0.0, **hline_settings)

    #============plot labels===============
    ax.set_xlabel(r"$\mathrm{Singles \, density} \, n_s$")
    ax.set_ylabel(r"$\xi \, \mathrm{(sites)}$")

    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    #============axes limits===============
    ax.set_xlim(1.0, 0.72)
    ax.set_ylim(-0.3, 5.5)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.2))

    ax.tick_params(pad=2)

    # ============second x-axis===============
    if x2 is not None:
        ax2 = ax.twiny()
        ax2.get_xaxis().tick_bottom()
        ax.get_xaxis().tick_top()
        ticks_bot = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25']
        ticks_top = [np.max(x)-float(tick)/x2 for tick in ticks_bot]

        # ============new labels===============
        ax.xaxis.set_label_coords(0.5, 1.13)
        ax2.set_xlabel(r"$\mathrm{Doping} \, \delta$")
        ax2.xaxis.set_label_coords(0.5, -0.12)

        # ============new axes limits===============
        ax2.set_xlim(ticks_top[0]+0.008, ticks_top[-1]-0.008)
        ax.set_xlim(ax2.get_xlim())

        # ============new ticks===============
        ax2.set_xticks(ticks_top)
        ax2.set_xticklabels(ticks_bot)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        # ax2.tick_params('x', bottom='off', length=0)

        ax2.tick_params(pad=2)
    else:
        ax2 = None

    return ax2


def plot_nn_corr(ax, dens, dens_err, corrx, corr, corr_err, far_idx=3, x2=None):

    x = [corrx[key] for key in corr.keys()][0]
    nn_corr = [corr[key][0] for key in corr.keys()]
    nn_corr_err = [corr_err[key][0] for key in corr.keys()]
    far_corr = [corr[key][far_idx] for key in corr.keys()]
    far_corr_err = [corr_err[key][far_idx] for key in corr.keys()]

    mec = color_blue_d
    mfc = color_blue_l
    ecolor = color_blue_d

    ax.errorbar(dens, nn_corr, xerr=dens_err, yerr=nn_corr_err,
                fmt="o", markeredgecolor=color_blue_d, markerfacecolor=color_blue_l, capsize=0, ecolor=color_blue_d,
                markeredgewidth=0.5, elinewidth=0.5, markersize=3.5, label=r"$d={:1.1f}$".format(x[0]), zorder=2)
    ax.errorbar(dens, far_corr, xerr=dens_err, yerr=far_corr_err,
                fmt="o", markeredgecolor=color_red_d, markerfacecolor=color_red_l, capsize=0, ecolor=color_red_d,
                markeredgewidth=0.5, elinewidth=0.5, markersize=3.5, label=r"$d={:1.1f}$".format(x[far_idx]),
                zorder=2)

    hline_settings = dict(linestyle='-', color='grey', zorder=0, linewidth=0.3, dashes=[1, 1])
    ax.axhline(y=0.0, **hline_settings)

    #============plot labels===============
    ax.set_xlabel(r"$\mathrm{Singles \, density} \, n_s$")
    ax.set_ylabel(r"$(-1)^i \, C_d$")

    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.12, 0.5)

    #============axes limits===============
    ax.set_xlim(1.0, 0.72)
    ax.set_ylim(-0.04, 0.36)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_minor_locator(MultipleLocator(0.02))

    ax.tick_params(pad=2)

    # ============second x-axis===============
    if x2 is not None:
        ax2 = ax.twiny()
        ax2.get_xaxis().tick_bottom()
        ax.get_xaxis().tick_top()
        ticks_bot = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25']
        ticks_top = [np.max(dens)-float(tick)/x2 for tick in ticks_bot]

        # ============new labels===============
        ax.xaxis.set_label_coords(0.5, 1.13)
        ax2.set_xlabel(r"$\mathrm{Doping} \, \delta$")
        ax2.xaxis.set_label_coords(0.5, -0.12)

        # ============new axes limits===============
        ax2.set_xlim(ticks_top[0]+0.008, ticks_top[-1]-0.008)
        ax.set_xlim(ax2.get_xlim())

        # ============new ticks===============
        ax2.set_xticks(ticks_top)
        ax2.set_xticklabels(ticks_bot)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        # ax2.tick_params('x', bottom='off', length=0)

        ax2.tick_params(pad=2)
    else:
        ax2 = None

    ax.legend(loc=(0.7, 0.8),  numpoints=1, borderpad=0.5, handlelength=0.4)

    return ax2


def plot_hh_corr(ax, x, x_err, corr, corr_err, x2=None):
    ax.errorbar(x, corr, yerr=corr_err, xerr=x_err,
                fmt="o", markeredgecolor=color_blue_d, markerfacecolor=color_blue_l, capsize=0, ecolor=color_blue_d,
                markeredgewidth=0.75, markersize=3.5, label=None, zorder=2)

    hline_settings = dict(linestyle='-', color='grey', zorder=0, linewidth=0.3, dashes=[1, 1])
    ax.axhline(y=0.0, **hline_settings)

    #============plot labels===============
    ax.set_xlabel(r"$\mathrm{Singles \, density} \, n_s$")
    ax.set_ylabel(r"$\langle hh \rangle_{NR}$")

    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.15, 0.5)

    #============axes limits===============
    ax.set_xlim(1.0, 0.80)
    ax.set_ylim(-0.052, 0.022)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.01))
    ax.yaxis.set_major_locator(MultipleLocator(0.01))
    ax.yaxis.set_minor_locator(MultipleLocator(0.002))

    ax.tick_params(pad=2)

    # ============second x-axis===============
    if x2 is not None:
        ax2 = ax.twiny()
        ax2.get_xaxis().tick_bottom()
        ax.get_xaxis().tick_top()
        ticks_bot = ['0.00', '0.05', '0.10', '0.15', '0.20', '0.25']
        ticks_top = [np.max(x)-float(tick)/x2 for tick in ticks_bot]

        # ============new labels===============
        ax.xaxis.set_label_coords(0.5, 1.13)
        ax2.set_xlabel(r"$\mathrm{Doping} \, \delta$")
        ax2.xaxis.set_label_coords(0.5, -0.12)

        # ============new axes limits===============
        ax2.set_xlim(ticks_top[0]+0.008, ticks_top[-1]-0.008)
        ax.set_xlim(ax2.get_xlim())

        # ============new ticks===============
        ax2.set_xticks(ticks_top)
        ax2.set_xticklabels(ticks_bot)
        ax.xaxis.set_major_locator(MultipleLocator(0.05))
        ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        # ax2.tick_params('x', bottom='off', length=0)

        ax2.tick_params(pad=2)
    else:
        ax2 = None

    return ax2


def plot_phasediagram(ax, arrowdir="vertical"):
    x1 = np.arange(0, 2, 0.1)
    y1 = -x1**2+4
    ax.fill_between(x1, 0, y1, facecolor=color_blue_l, linewidth=0)

    x2 = np.arange(3, 9, 0.1)
    y2 = 0.3*np.sqrt(9-(x2-6)**2)-0.3
    ax.fill_between(x2, 0, y2, facecolor=color_red_l, linewidth=0)

    if arrowdir == 'vertical':
        ax.arrow(0.05, 5.2, 0.0, -3.2, head_width=0.35, head_length=0.3, fc=color_red_m, ec=color_red_m,
                 linewidth=1.7, clip_on=False, zorder=4)
        ax.arrow(0.05*4.0, 2*0.95+3.2, 0.0, -3.2, head_width=0.35, head_length=0.3, fc='w', ec='w',
                 linewidth=1.7, clip_on=False, zorder=3)
    elif arrowdir == 'horizontal':
        ax.arrow(0.17, 2, 3, 0, head_width=0.2, head_length=0.5, fc=color_red_m, ec=color_red_m,
                 linewidth=1.7, clip_on=False, zorder=4)
        ax.arrow(0.17, 2*0.95, 3, 0, head_width=0.2, head_length=0.5, fc='w', ec='w',
                 linewidth=1.7, clip_on=False, zorder=3)

    #============plot labels===============
    # if arrowdir == 'horizontal':
    ax.annotate("Density", xy=(4.8, -0.9), xycoords='data', fontsize=5, annotation_clip=False)
    ax.annotate("Temperature", xy=(-1.6, 5.1), xycoords='data', fontsize=5, annotation_clip=False, rotation=90)
    # else:
    #     ax.annotate("Density", xy=(5.5, -0.8), xycoords='data', fontsize=5, annotation_clip=False)
    #     ax.annotate("Temperature", xy=(-1.4, 5.2), xycoords='data', fontsize=5, annotation_clip=False, rotation=90)

    #============axes limits===============
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)

    #============ticks===============
    plt.sca(ax)
    frame = plt.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)

    return


def plot_dens_map(ax, dens):

    cm = brewer2mpl.get_map('Purples', 'Sequential', 8).mpl_colormap
    ax.pcolormesh(dens, norm=colors.PowerNorm(gamma=1./2.), vmin=0, vmax=1, cmap=cm, rasterized=True)

    circ = plt.Circle((60.5, 57.5), 5, color='w', fill=False, clip_on=False, ls=(0, (1, 1)))
    ax.add_artist(circ)

    # ax.axhline(y=(30+85)/2.0)
    # ax.axvline(x=(33+88)/2.0)

    # ============plot labels===============
    ax.set_xlabel(r"$r \, \mathrm{sites}$")
    ax.set_ylabel(r"$r \, \mathrm{sites}$")

    ax.xaxis.set_label_coords(0.5, -0.06)
    ax.yaxis.set_label_coords(-0.06, 0.5)

    # ============axes limits===============
    ax.set_xlim(33, 88)
    ax.set_ylim(30, 85)

    # ============ticks===============
    plt.sca(ax)
    plt.xticks([60.5-20, 60.5, 60.5+20], [-20, 0, 20])
    plt.yticks([57.5-20, 57.5, 57.5+20], [-20, 0, 20])

    ax.tick_params(pad=2)

    return cm


def plot_dens_prof(ax, rad, prof, err):

    rad = np.concatenate((-1*rad[::-1], rad[:]))
    prof = np.concatenate((prof[::-1], prof[:]))
    err = np.concatenate((err[::-1], err[:]))

    ax.errorbar(rad, prof, yerr=err,
                fmt="o", markeredgecolor=color_blue_d, markerfacecolor=color_blue_l, capsize=0, ecolor=color_blue_d,
                markeredgewidth=0.75, markersize=3.5, label=None, zorder=2)

    hline_settings = dict(linestyle='-', color='grey', zorder=0, linewidth=0.3, dashes=[1, 1])
    ax.axhline(y=0, **hline_settings)

    vline_settings = dict(linestyle='-', color='black', zorder=0, linewidth=0.3, dashes=[1, 1])
    ax.axvline(x=-5, **vline_settings)
    ax.axvline(x=5, **vline_settings)

    # ============plot labels===============
    ax.set_xlabel(r"$r \, \mathrm{sites}$")
    ax.set_ylabel(r"$n_s$")

    ax.xaxis.set_label_coords(0.5, -0.12)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    # ============axes limits===============
    ax.set_xlim(-11, 11)
    ax.set_ylim(-0.05, 1.05)

    # ============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(5))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.2))
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    ax.tick_params(pad=2)

    return


def plot_dens_prof_zoom(ax, rads, profs, errs, densavg, doping):

    cmap = brewer2mpl.get_map('Blues', 'Sequential', 8).mpl_colormap
    # order = [3, 4, 1, 2, 7, 0, 5, 6]
    # order = [0, 1, 2, 3, 4, 5, 6, 7]
    order = [5, 1, 3, 0, 2, 6, 7, 4]
    i = 0
    hh = []
    for key in rads.keys():
        if order[i] % 3 == 0:
            rad = np.concatenate((-1*rads[key][::-1], rads[key][:]))
            prof = np.concatenate((profs[key][::-1], profs[key][:]))
            err = np.concatenate((errs[key][::-1], errs[key][:]))

            # print order[i], np.mean(prof[15:-15])
            j = 0.99 - order[i] * 0.07

            hh.append(ax.errorbar(rad, prof, yerr=err,
                                  fmt="o", markeredgecolor=cmap(j), markerfacecolor=cmap(j-0.05), capsize=0,
                                  ecolor=cmap(j), markeredgewidth=0.75, markersize=3.5,
                                  label=r"$\delta = {:0.3f}$".format(doping[order[i]]), zorder=2))

            hline_settings = dict(linestyle='-', color=cmap(j), zorder=0, linewidth=0.3)#, dashes=[1, 1])
            ax.axhline(y=densavg[order[i]], **hline_settings)
            # ax.axhline(y=np.mean(prof[15:-15]), **hline_settings)

        # print densavg[i], np.sum(2*np.pi*rads[key][:5]*profs[key][:5])/np.sum(2*np.pi*rads[key][:5])
        # print np.sqrt(np.sum((rads[key][:5]*errs[key][:5])**2))/np.sum(rads[key][:5])
        i += 1

    # ============plot labels===============
    ax.set_xlabel(r"$r \, \mathrm{sites}$")
    ax.set_ylabel(r"$n_s$")

    ax.xaxis.set_label_coords(0.5, -0.04)
    ax.yaxis.set_label_coords(-0.08, 0.5)

    # ============axes limits===============
    ax.set_xlim(-5, 5)
    ax.set_ylim(0.68, 0.97)

    # ============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.yaxis.set_major_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.01))

    ax.tick_params(pad=2)

    ax.legend([hh[1], hh[0], hh[2]], [hh[1].get_label(), hh[0].get_label(), hh[2].get_label()],
              loc=(0.03, 0.03),  numpoints=1, borderpad=0.5, handlelength=0.4)

    return


def make_histogram(ax, data, theory, **kwargs):
    """
    make histogram based on the unbinned data data
    :param data:
    :param kwargs:
    :return:
    """

    print (min(data), max(data))

    n_bins = kwargs.pop('n_bins', 21)
    hist_type = kwargs.pop('hist_type', 'stepfilled')
    is_normed = kwargs.pop('is_normed', True)
    vlines = kwargs.pop('vlines', [])
    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    xlim = kwargs.pop('xlim', (0, 10))
    ylim = kwargs.pop('ylim', (0, 10))
    title = kwargs.pop('title', None)
    major_tick_dx = kwargs.pop('major_tick_dx', 5)
    major_tick_dy = kwargs.pop('major_tick_dy', 5)
    minor_tick_dx = kwargs.pop('minor_tick_dx', 1)
    minor_tick_dy = kwargs.pop('minor_tick_dy', 1)
    color = kwargs.pop('color', 'blue')

    majorLocatorX = MultipleLocator(major_tick_dx)
    majorLocatorY = MultipleLocator(major_tick_dy)
    minorLocatorX = MultipleLocator(minor_tick_dx)
    minorLocatorY = MultipleLocator(minor_tick_dy)

    if color == 'blue':
        edge_color = color_blue_d
        center_color = color_blue_l
    elif color == 'red':
        edge_color = color_red_d
        center_color = color_red_l
    elif color == 'teal':
        edge_color = color_teal_d
        center_color = color_teal_l
    elif color == 'brown':
        edge_color = color_brown_d
        center_color = color_brown_l

    theory[2] = theory[2]/(np.sum(theory[2])*(theory[1][1]-theory[1][0]))
    # ax.bar(theory[0], theory[2], width=theory[1][0]-theory[0][0],
    #        color=(1, 1, 1, 0), edgecolor='k', zorder=2)

    n, bins, _ = ax.hist(data, bins=n_bins, range=xlim, edgecolor=edge_color, color=center_color, histtype=hist_type,
                         normed=is_normed, label='Data')
    # ax.errorbar(bins[:-1]+(bins[0]+bins[1])/2., n, yerr=np.sqrt(n)/(np.sum(data)*(bins[1]-bins[0])),
    #             fmt="o", markeredgecolor=edge_color, markerfacecolor=center_color, capsize=0, ecolor=edge_color,
    #             markeredgewidth=0, markersize=0, label=None, zorder=2)

    x = [2*val for pair in zip(theory[0], theory[1]) for val in pair]
    y = [0.5*val for pair in zip(theory[2], theory[2]) for val in pair] #0.5 ensures normalization

    ax.plot(x, y, color='k', linewidth=0.5, label='Heisenberg QMC', zorder=2)

    ax.xaxis.set_major_locator(majorLocatorX)
    ax.xaxis.set_minor_locator(minorLocatorX)

    ax.yaxis.set_major_locator(majorLocatorY)
    ax.yaxis.set_minor_locator(minorLocatorY)

    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=4, fontsize=5.5)
    else:
        ax.set_xticklabels([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=4, fontsize=5.5)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    else:
        ax.set_xticklabels([])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title is not None:
        plt.title(title, fontsize=7)

    if vlines:
        vline_settings = dict(linestyle='-', color='grey', zorder=5, linewidth=0.3, dashes=[1, 1])
        for vline in vlines:
            ax.axvline(vline, **vline_settings)


def plot_hist_theory(ax, th0, th1, th2, **kwargs):

    xlabel = kwargs.pop('xlabel', None)
    ylabel = kwargs.pop('ylabel', None)
    xlim = kwargs.pop('xlim', (0, 10))
    ylim = kwargs.pop('ylim', (0, 10))
    title = kwargs.pop('title', None)
    major_tick_dx = kwargs.pop('major_tick_dx', 5)
    major_tick_dy = kwargs.pop('major_tick_dy', 5)
    minor_tick_dx = kwargs.pop('minor_tick_dx', 1)
    minor_tick_dy = kwargs.pop('minor_tick_dy', 1)

    majorLocatorX = MultipleLocator(major_tick_dx)
    majorLocatorY = MultipleLocator(major_tick_dy)
    minorLocatorX = MultipleLocator(minor_tick_dx)
    minorLocatorY = MultipleLocator(minor_tick_dy)

    th0[2] = th0[2]/(np.sum(th0[2])*(th0[1][1]-th0[1][0]))
    x0 = [2*val for pair in zip(th0[0], th0[1]) for val in pair]
    y0 = [0.5*val for pair in zip(th0[2], th0[2]) for val in pair] # 0.5 ensures normalization
    ax.plot(x0, y0, color='k', linewidth=0.5, label='Heisenberg QMC', zorder=2)

    # x1 = th1[0]
    # y1 = 2*th1[1]/(np.sum(th1[1])*(th1[0][1]-th1[0][0]))
    #print list(2*th0[0])+[2*th0[1][-1]]
    #print th1[0]
    y1, _ = np.histogram(th1[0], bins=list(2*th0[0])+[2*th0[1][-1]], weights=th1[1], density=True)
    y1 = [val for pair in zip(y1, y1) for val in pair]
    ax.plot(x0, y1, color=color_blue_m, linewidth=0.5, label='Classical Heisenberg MC', zorder=2)

    ax.xaxis.set_major_locator(majorLocatorX)
    ax.xaxis.set_minor_locator(minorLocatorX)

    ax.yaxis.set_major_locator(majorLocatorY)
    ax.yaxis.set_minor_locator(minorLocatorY)

    if xlabel is not None:
        ax.set_xlabel(xlabel, labelpad=4, fontsize=5.5)
    else:
        ax.set_xticklabels([])
    if ylabel is not None:
        ax.set_ylabel(ylabel, labelpad=4, fontsize=5.5)
        ax.yaxis.set_label_coords(-0.1, 0.5)
    else:
        ax.set_yticklabels([])
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if title is not None:
        plt.title(title, fontsize=7)

    return


def plot_stability(ax, dat, labels=(True, True), ylabel=None):
    for d in dat:
        # Get x and y data, then pick out the no spin removal data
        x = d[0][0]
        y = d[1][0]
        ax.errorbar(x, y,
                    fmt="o", markeredgecolor=color_blue_d, markerfacecolor=color_blue_l, capsize=0, ecolor=color_blue_d,
                    markeredgewidth=0.75, markersize=3.5, label=None, zorder=2)

    if ylabel == "Total atom number":
        x = range(1501)
        bound = [420 if i < 600 or i > 1400 else 350 for i in x]
        ax.fill_between(x, 0, bound, color=color_grey_l)
    else:
        ax.axvspan(-10, 600, color=color_grey_l)
        ax.axvspan(1400, 1510, color=color_grey_l)

    #============plot labels===============
    ax.set_xlabel("Image number")
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    ax.xaxis.set_label_coords(0.5, -0.08)
    ax.yaxis.set_label_coords(-0.035, 0.5)

    #============axes limits===============
    ax.set_xlim(-1, 1501)
    if ylabel == "Total atom number":
        ax.set_ylim(300, 400)
    else:
        ax.set_ylim(-8, 8)

    #============ticks===============
    ax.xaxis.set_major_locator(MultipleLocator(200))
    ax.xaxis.set_minor_locator(MultipleLocator(50))

    if ylabel == "Total atom number":
        ax.yaxis.set_major_locator(MultipleLocator(20))
        ax.yaxis.set_minor_locator(MultipleLocator(5))
    else:
        ax.yaxis.set_major_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

    ax.tick_params(pad=2)

    return


def get_color(cmap, (cs, cf, cpow), num_c, i):
    if num_c == 1:
        return cmap(1)
    else:
        beta = (cf-cs)/(num_c-1.0)**cpow
        return cmap(cs+beta*i**cpow)


def plot_only_data(plt, data_x, data_y, err_x, err_y, (cs, cf, cpow),
        xlim=None, ylim=None, ymax=None, label="", tick_format='%1.2f',xy_text=(-15,0),
        trunc_marg=None,cmap=cmap,th_x2=None,th_y2=None, is_logy = False, data_label=None, markers=None):
    lines = []
    points = []

    if trunc_marg != None and xlim != None:
        xmin = min(xlim)
        xmax = max(xlim)
        ol = [len(l) for l in data_x]
        data_x, data_y, err_x, err_y = zip(*[zip(*[v for v in zip(*t) if
                v[0]-trunc_marg > xmin and v[0]+trunc_marg < xmax]) for t in
                zip(data_x, data_y, err_x, err_y)])
        trunc = [ol[i]-len(l) for i, l in enumerate(data_x)]
        if sum(trunc) > 0:
            print "warning! points are being truncated off the graph"
            print "truncated from:" + str([(ol[i], len(l)) for i, l in enumerate(data_x)])

    # color defined as cmap((cf*i+cs)/ct)
    num_c = len(data_x)
    light_fac = 1 # smaller is darker

    err_x = np.array(err_x)
    print err_y
    err_y = np.array(err_y)

    for i in range(num_c):
        idx = [0, 4, 5, 6]
        cd = get_color(cmap, (cs, cf, cpow), num_c+5, idx[i])
        cl = tuple([(x+light_fac)/(1+light_fac) for x in list(cd)])

        if ymax != None and len(err_y[i].shape) != 2:
            err_y_up = [min(data_y[i][j]+err_y[i][j], ymax)-data_y[i][j] for j in
                    range(len(data_y[i]))]
        else:
            err_y_up = err_y[i]

        err_x[i] = np.array(err_x[i])
        err_y[i] = np.array(err_y[i])
        print err_y[i]

        pars = {'linewidth':1,
                'fmt':'o',
                'ms':ms,
                'mew':mew,
                'c':cd[:-1]+(0.6,),
                'mec':cd,
                'mfc':cl,
                'capsize':0,
                'zorder':.2*i+.1}
        if data_label is not None:
            pars['label']=data_label[i]
        if markers is not None:
            pars['marker']=markers[i]
        pars['xerr'] = (err_x[i] if len(err_x[i].shape)==1 else np.transpose(err_x[i]))
        pars['yerr'] = ([err_y[i], err_y_up] if len(err_y[i].shape)==1 else np.transpose(err_y[i]))

        p, q, r, = plt.errorbar(data_x[i], data_y[i], **pars)

        points += [(p)]

    if xlim != None: plt.set_xlim(xlim)
    if ylim != None: plt.set_ylim(ylim)
    plt.xaxis.set_tick_params(size=2)
    plt.yaxis.set_tick_params(size=2)
    text = plt.annotate(label, xy=(0,1), xycoords='axes fraction',
            xytext=xy_text, textcoords='offset points',
            weight='bold', backgroundcolor='white', zorder=.1,
            ha='left', va='top', size=9)
    bbox = text.get_bbox_patch()
    #bbox.set_boxstyle('square', pad=0.1)
    plt.tick_params('x', pad=2)
    plt.tick_params('y', pad=2)
    plt.yaxis.set_major_formatter(mtick.FormatStrFormatter(tick_format))
    if is_logy:
        print "setting logy"
        plt.set_yscale("log", nonposy='clip')

    return lines, points

