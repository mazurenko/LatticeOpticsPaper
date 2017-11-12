import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import pylab as pl
from plot_tools import *
import matplotlib as mpl
import brewer2mpl as cb
from matplotlib import rcParams


def berek(theta0, theta1, phi):
    return np.square(np.absolute(np.exp(1j*phi)*np.cos(2*theta0)*np.cos(2*theta1) - np.sin(2*theta0)*np.sin(2*theta1)))


def generate_berek_scan(retardation_range=(0, 2*np.pi),
                        theta_angles=[0, np.pi/4/8, np.pi/16, np.pi*3/4/8, np.pi/8],
                        n_phi=1e3):
    phi = np.linspace(retardation_range[0], retardation_range[1], n_phi)
    return phi, [berek(theta, theta, phi) for theta in theta_angles]


if __name__ == "__main__":
    labeled_angles = [(r"\pi/32", np.pi/32), (r"\pi/16", np.pi/16), (r"3\pi/32", 3*np.pi/32), (r"\pi/8", np.pi/8)]
    angle_labels, angles = zip(*labeled_angles)
    phi, intensities = generate_berek_scan(theta_angles=angles)

    rcParams.update({})
    scl = 1/1.61
    fig_width = 3.37
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
    w_a = 0.10    # width of a) figures
    h_a = w_a/scl  # height of a) figures
    ow_a = w_a + 0.015
    oh_a = h_a + 0.068
    ow_a0 = 0.065
    oh_a0 = 0.22
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('$\phi$')
    ax.set_ylabel('$I/I_0$')
    ax.set_xlim((0, np.max(phi)))
    ax.set_ylim((0, 1.39))

    for idx, intensity in enumerate(intensities):
        ax.plot(phi, intensity, label=r"$\theta={}$".format(angle_labels[idx]), zorder=idx,
                color= cb.get_map('OrRd', 'sequential', len(intensities)+2).mpl_colors[2+idx])
    ax.axhline(1, linewidth=1, color=color_grey, zorder=-1)
    ax.legend()
    plt.savefig('berek_sim.pdf')
