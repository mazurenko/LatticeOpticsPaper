import os
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import pylab as pl
from plot_tools import *
import matplotlib as mpl
import brewer2mpl as cb
from matplotlib import rcParams

def fourier_ser(ser, n_pad=10000, is_window=True, time_shift=0):
    """
    performs a fourier transform in the context of the pandas package
    :param ser: time series input data. The index is assumed to be the time
    :param n_pad: how much to pad the array (edge mode)
    :param is_window: whether to apply the blackman window
    :param time_shift: by convention, fft assumes t=0 at the first sample point. If you want t=0 to be referenced to a
    different time (e.g. when working with impulse or step responses), timeshift let's you set that time. Only affects phases
    :return: series of frequencies (index) to complex vals
    """
    data = np.pad(ser.values, (n_pad, n_pad), 'edge')  # pad to deal with fft artifacts from finite sampling time
    n = data.size
    window = np.blackman(n)
    data = np.multiply(window, data)
    fourier = np.fft.fftshift(np.fft.fft(data))
    timestep = np.mean(ser.index[1:].values - ser.index[0:-1].values)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=timestep))
    df = pd.DataFrame(data=np.transpose(np.vstack((np.abs(fourier),
                                                     (np.angle(fourier) + 2*np.pi*freqs*(time_shift+timestep*n_pad) + np.pi) % (2*np.pi) - np.pi))),
                        index=freqs, columns=['amp', 'angle'])
    #print df[-20000:20000]
    return df

def get_transfer_function(filename, data_dir='data'):
    df = pd.read_csv(os.path.join(data_dir, filename), delimiter=',', header=2, names=['time', 'setpoint', 'response'])
    # The scope occasionally outputs garbage for the first datapoint, that's why that line is skipped by header = 2
    df = df.set_index('time')
    df -= df.setpoint.min()
    ft_setpoint = fourier_ser(df['setpoint'], n_pad=10000)
    ft_response = fourier_ser(df['response'], n_pad=10000)
    transfer_function = ft_response
    transfer_function.amp /= ft_setpoint.amp
    transfer_function.angle -= ft_setpoint.angle
    transfer_function.angle = (transfer_function.angle + np.pi) % (2*np.pi) - np.pi
    return transfer_function


def plot_transfer(filename):
    window = [100, 12000]  # in units of Hz
    #tf = get_transfer_function('scope_164.csv')
    tf = get_transfer_function(filename)
    tf = tf[window[0]:window[1]]
    axes = tf.plot(subplots=True, legend=False)#, logx=True)
    axes[1].set_xlabel('Frequency (Hz)')
    plt.show()


def fit_transfer_function(tf, freq_window=[10, 12000], guess=[3800, .25, 5000], order='second'):
    """
    Fits the transfer function to the second order form
    :param tf: transfer function dataframe, with frequency as index, and columns amp and angle
    :param freq_window: frequency window in which to fit - list of 2 elements [low, high]
    :param guess: guess for the fit parameters - list of 2 elements [f0, xi]
    :param order: what order to fit
    Fits the transfer function to a second order of the form H(w) = (1-(w/w0)^2 + 2*j*xi*w/w0)^-1
    :return: fit parameters
    """
    assert order in ['second', 'third']

    xdata = tf[freq_window[0]:freq_window[1]].index.values
    ydata = tf[freq_window[0]:freq_window[1]].amp.values*np.exp(1j*tf[freq_window[0]:freq_window[1]].angle.values)
    fake_ydata = np.real(np.zeros_like(ydata))  # curve_fit fails over imaginary numbers - hence the np.real( ) call
    print fake_ydata

    #  Note - there is a bit of a hack here since the curve fit does not handle imaginary numbers nicely, so we are
    #  fitting to an array of zeros, and actually inserting the data into the fit function
    def transfer_second_order(f, f0, xi):
        return np.reciprocal(1-np.square(np.divide(f, f0)) + 2*1j*np.multiply(xi, np.divide(f, f0)))

    def transfer_third_order(f, f0, xi, alpha):
        return np.reciprocal(np.multiply(1-np.square(np.divide(f, f0)) + 2*1j*np.multiply(xi, np.divide(f, f0)),
                                         1j*np.divide(f, alpha)+1))

    fun_dict = {'second': transfer_second_order, 'third': transfer_third_order}
    guess = guess if order == 'third' else guess[:2]
    tfun = fun_dict[order]
        #def fit_fun(f, f0, xi, alpha):
        #    return np.abs(np.subtract(transfer_third_order(f, f0, xi, alpha), ydata))
        #popt, perr = curve_fit(fit_fun, xdata, fake_ydata, p0=guess)

    def fit_fun(*args):
        return np.abs(np.subtract(tfun(*args), ydata))
    popt, perr = curve_fit(fit_fun, xdata, fake_ydata, p0=guess)

    args = [xdata]
    args.extend(popt)
    plt.plot(xdata, np.abs(tfun(*args)), marker='o', ms=2)
    plt.plot(xdata, np.abs(ydata), marker='o', ms=2)
    plt.show()
    return popt, perr


def guess_parabola_params(x0, x1, x2, y1, y2):
    """
    Guess the parameters of the parabola of the form y = amp*(x-a)(x-x0), where its known that it passes through the points
    (x0, 0), (x1, y1), (x2, y2)
    :param x0:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :return: amp, a, x0
    """
    amp = (-x0*y1 + x2*y1 + x0*y2 - x1*y2) / ((x1 - x2)*(x0**2 - x0*x1 - x0*x2 + x1*x2))
    a = (x0*x2*y1 - x2**2*y1 - x0*x1*y2 + x1**2*y2) / (x0*y1 - x2*y1 - x0*y2 + x1*y2)
    return amp, a, x0


def parabola_zero(x, amp, a, x0):
    """
    fitting function for parabpla passing near zero
    :return:
    y = amp*(x-a)(x-x0)
    """
    return np.multiply(np.multiply(amp, x-a), x-x0)


def cross(series, cross=0, direction='cross', lower_bound=None):
    """
    Taken from Stackexchange user dailyglen
    Given a Series returns all the index values where the data values equal
    the 'cross' value.

    Direction can be 'rising' (for rising edge), 'falling' (for only falling
    edge), or 'cross' for both edges
    """
    # Find if values are above or bellow yvalue crossing:
    if lower_bound is not None:
        series = series[lower_bound:]
    above = series.values > cross
    below = np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]

    # Calculate x crossings with interpolation using formula for a line:
    x1 = series.index.values[idxs]
    x2 = series.index.values[idxs+1]
    y1 = series.values[idxs]
    y2 = series.values[idxs+1]
    x_crossings = (cross-y1)*(x2-x1)/(y2-y1) + x1

    return x_crossings


def find_value(am, target, fit_window=500, direction='cross', lower_bound = 1000,  is_plot=False):
    """
    Finds closest location to target - estimate the location by reading off the nearest value,
    take a fit_window width window around it and fit to a second order polynomial
    :param am:
    :param target:
    :param fit_window:
    :return: (cross_value, error)
    """

    try:
        guess = cross(am, cross=target, direction=direction, lower_bound=lower_bound)[0]
    except IndexError:
        print "index error happened, skip point"
        return None, None
    print "guess"
    print guess
    am = am[guess-fit_window/2:guess+fit_window/2] - target
    args = guess, am.index.values[0], am.index.values[-1], am.values[0], am.values[-1]
    guess_params = guess_parabola_params(*args)
    popt, pcov = curve_fit(parabola_zero, am.index.values, am.values, p0=guess_params)
    perr = np.sqrt(np.diag(pcov))

    if is_plot:
        x = np.linspace(am.index.values[0], am.index.values[-1])
        y = parabola_zero(x, popt[0], popt[1], popt[2])
        ax = am.plot()
        ax.plot(x, y)
        plt.show()
    return popt[2], perr[2]


def find_rolloff(tf, freq_window=[10, 15000], fit_window=500, is_plot=False):
    """
    Finds the 3dB and pi phase shift points. To find 3dB, estimate the location by reading off the nearest value,
    take a 1 kHz window around it and fit to a second order polynomial.
    To find pi point, unroll the phase and do same procedure
    :param tf: transfer function dataframe
    :param freq_window: window where to search
    :return: three_dB, pi_phase_shift - each is a 2 length array with form (value, error)
    """
    tf_temp = tf[freq_window[0]:freq_window[1]]
    if is_plot:
        tf_temp.plot()
        plt.show()

    #  find_3db
    three_db = find_value(tf_temp.amp, 0.5)

    #  find pi phase shift
    tf_temp.angle = (tf_temp.angle + 2*np.pi) % (2*np.pi) - 2*np.pi
    pi_phase_shift = find_value(tf_temp.angle, -1*np.pi)
    return {'three_dB_pt': three_db[0], 'three_dB_pt_err': three_db[1],
            'pi_phase_point': pi_phase_shift[0], 'pi_phase_point_err': pi_phase_shift[1]}


def get_tfun_parameters(data_id_filename, column, is_plot=False):
    """
    Extract transfer function parameters from files indicated in the given column in given data_id_filename
    :param data_id_filename:
    :param column:
    :return:
    """
    file_df = pd.read_csv(os.path.join('data', data_id_filename))
    file_df = file_df.set_index('Setpoint_mW')
    loop_values = {}
    for idx, row in file_df.iterrows():
        f = row[column]
        tf = get_transfer_function(f)
        if is_plot:
            tf[0:14000].plot()
            plt.show()
        rolloff_parameters = find_rolloff(tf)
        loop_values[idx] = rolloff_parameters
        print f
        print rolloff_parameters
    loop_df = pd.DataFrame.from_dict(loop_values, orient='index')
    loop_df.index.name = 'power'
    return loop_df

if __name__ == "__main__":
    #print get_tfun_parameters('data_id_lp.csv', 'Measurement_NE')
    nw = get_tfun_parameters('data_id_lp.csv', 'Measurement_NW', is_plot=False)
    nw.to_csv(os.path.join('results', 'nw_results.csv'))

    ne = get_tfun_parameters('data_id_lp.csv', 'Measurement_NE', is_plot=False)
    ne.to_csv(os.path.join('results', 'ne_results.csv'))
