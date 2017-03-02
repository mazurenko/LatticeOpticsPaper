import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl as cb


def plot_cameras(datevec, **kwargs):
    base_directory = kwargs.pop('base_directory', 'W:Runlog')
    stability_scan_dir = kwargs.pop('stability_scan_dir', 'stability_scan')
    day_dir = '%s\\%04d\\%02d\\%02d' % (base_directory, datevec[0], datevec[1], datevec[2])
    stability_path = "%s\\%s" % (day_dir, stability_scan_dir)
    ne = 'out20160916 1948-1.csv'
    nufern = 'out20160916 1948-2.csv'
    files_to_analyze = [ne, nufern]

    data = []
    for filename in files_to_analyze:
        data.append(load_data("%s\\%s" % (stability_path, filename)))

    fig = plt.figure(figsize=(10, 12))
    ax_ne_x = fig.add_subplot(411)
    ax_ne_y = fig.add_subplot(412)
    ax_nufern_x = fig.add_subplot(413)
    ax_nufern_y = fig.add_subplot(414)

    colors = cb.get_map('Dark2', "Qualitative", 4).mpl_colors
    plot_data(ax_ne_x, data[0][:, 0], data[0][:, 1], ylabel='NE x (pix)', color=colors[0])
    plot_data(ax_ne_y, data[0][:, 0], data[0][:, 2], ylabel='NE y (pix)', color=colors[1])
    plot_data(ax_nufern_x, data[1][:, 0], data[1][:, 1], ylabel='Nufern x (pix)', color=colors[2])
    plot_data(ax_nufern_y, data[1][:, 0], data[1][:, 2], ylabel='Nufern y (pix)', xlabel="Time (min)", color=colors[3])
    #plt.savefig("%s\\compiled_plot.png" % stability_path)
    plt.show()


def plot_data(ax, x, y, xlabel=None, ylabel=None, ylim = [-1.5, 1.5], color = None):
    param_dict = {}
    if color is not None:
        param_dict['color'] = color
    ax.plot((x - np.min(x))/60.0, y - np.mean(y), **param_dict)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlabel is not None:
        ax.set_xlabel(xlabel)


def load_data(filename):
    return np.genfromtxt(filename, delimiter=',')


if __name__ == "__main__":
    plot_cameras((2016, 9, 16), base_directory='Y:Runlog')
