import numpy as np
import pyfits
import os
import matplotlib.pyplot as plt
import plastia.analysis.fit as pfit # internal Greinerlab fitting suite
import pandas as pd

data_dir = "data"

"""
First axis always the "narrow" axis of the cloud for nw beam
"""


def running_mean(x, n=20):
    return np.convolve(x, np.ones((n,)) / n, mode='valid')


def load_pointing(name):
    """
    Get the pointing data from the data subdir
    :param name:
    :return:
    """
    pointing_data = np.loadtxt(os.path.join(data_dir, '{}.csv'.format(name)), delimiter=',')[:, 1:3]
    return pointing_data


def load_atom_matrixes(name):
    """
    Get the atom matrixes from the data subdir
    :param name:
    :return:
    """
    hdulist = pyfits.open(os.path.join(data_dir, '{}.fits'.format(name)))#, ignore_missing_end=True)
    data_list = [hdulist[i].data for i in range(1, len(hdulist))]
    return data_list


def get_fitted_centers(matrix_list):
    centroids = []
    for matrix in matrix_list:
        fit = pfit.FitFunctionGaussian2D()
        fit.guess_matrix(matrix)
        fit.fit_matrix(matrix)
        print fit.result.parameters
        print fit.result
        centroids.append(map(lambda x: x.bare_value, fit.result.parameters)[2:4][::-1])
    return reduce(lambda x, y: np.vstack((x, y)), centroids)


def get_centroids(matrix_list, is_plot =False):
    if is_plot:
        plt.imshow(matrix_list[0])
        plt.show()
    centroids = [np.divide(np.argwhere(mat == 1).sum(0), np.sum(mat)) for mat in matrix_list]
    stdvs = [np.std(np.argwhere(mat == 1), axis=0) for mat in matrix_list]
    return reduce(lambda x, y: np.vstack((x, y)), centroids)


def affine_fit(p0, p1):
    """
    finds an affine fit between p0 and p1
    :param p0: N by 2 array representing points in plane 0
    :param p1: N by 2 array representing points in plane 1
    :return: an affine fit matrix
    """
    assert np.shape(p0) == np.shape(p1)
    n_data, n_dim = np.shape(p0)
    p0 = np.hstack((p0, np.ones((n_data, 1))))
    x, res, rank, s = np.linalg.lstsq(p0, p1)
    return x


def affine_transform(trans_mat, p0):
    """
    transforms a vector using an affine transform
    :param trans_mat: matrix transform
    :param p0: N by 2 array of points to transform
    :return:
    """
    n_data, n_dim = np.shape(p0)
    p0 = np.hstack((p0, np.ones((n_data, 1))))
    #return np.transpose(np.dot(np.transpose(trans_mat), np.transpose(p0)))
    return np.dot(p0, trans_mat)


def get_affine_calibration(filename='calibration', bad_points=(), is_plot=False, is_fit=False, is_save=True):
    atom_matrixes = load_atom_matrixes(filename)

    centroids = get_fitted_centers(atom_matrixes) if is_fit else get_centroids(atom_matrixes)
    pointings = load_pointing(filename)

    # filter bad fits, if any
    centroids = np.delete(centroids, bad_points, axis=0)
    pointings = np.delete(pointings, bad_points, axis=0)

    transfer_matrix = affine_fit(pointings, centroids)
    p1 = affine_transform(transfer_matrix, pointings)

    if is_plot:
        plt.plot(centroids[:, 0], centroids[:, 1], linestyle='none', marker='o')
        plt.plot(p1[:, 0], p1[:, 1], linestyle='none', marker='o')
        plt.show()

    comparison_data = pd.DataFrame(np.hstack((centroids, p1)), columns=["x_t", "x_l", "x_ct", "x_cl"])
    comparison_data.to_csv('calibration_data_comparison.csv')
    return transfer_matrix, comparison_data


def compare_out_of_sample(transfer_matrix, filename='stability_check', is_fit=False):
    atom_matrixes = load_atom_matrixes(filename)
    centroids = get_fitted_centers(atom_matrixes) if is_fit else get_centroids(atom_matrixes)
    pointings = load_pointing(filename)
    p1 = affine_transform(transfer_matrix, pointings)
    comp_data = np.hstack((centroids, p1))
    return comp_data


def split_into_batches(elements, period_size, sample_size):
    """
    splits the list of elements into list of lists of periods "period"-"sample"-"period"...-"remainder"
    :param elements:
    :param period_size:
    :param sample_size:
    :return:
    """
    elements.reverse()
    batches = []
    current_batch = []
    target_counter = 0
    is_period_target = True
    while elements:
        target = period_size if is_period_target else sample_size
        e = elements.pop()
        current_batch.append(e)
        target_counter += 1
        if target_counter == target:
            target_counter = 0
            is_period_target = not is_period_target
            batches.append(current_batch)
            current_batch = []
    if current_batch:
        batches.append(current_batch)
    return batches


def compare_out_of_sample_with_updating(transfer_matrix, filename='stability_check', is_fit=False, is_save=True,
                                        update_period=60,
                                        update_sample=6):
    """
    Given initial transfer matrix, compare the transformed measurements to the actual measurements, updating the offset
    every update_period, based on the update_samples
    :param transfer_matrix:
    :param filename:
    :param is_fit:
    :param update_period:
    :param update_sample:
    :return:
    """
    atom_matrixes = load_atom_matrixes(filename)
    centroids = get_fitted_centers(atom_matrixes) if is_fit else get_centroids(atom_matrixes)
    pointings = load_pointing(filename)
    batches = split_into_batches(zip(list(centroids), list(pointings)), update_period, update_sample)
    oos_batches = batches[::2]
    ins_batches = batches[1::2]

    positional_trace = []
    while oos_batches or ins_batches:
        if oos_batches:
            # calculate out of sample batch
            oos_batch = oos_batches.pop(0)
            ss, cam = map(np.vstack, zip(*oos_batch))
            transformed_cam = affine_transform(transfer_matrix, cam)
            positional_trace.append(np.hstack((ss, transformed_cam)))
        if ins_batches:
            # look at small in-sample batch, correct transfer matrix
            ins_batch = ins_batches.pop(0)
            ss, cam = map(np.vstack, zip(*ins_batch))
            transformed_cam = affine_transform(transfer_matrix, cam)
            avg_error = np.mean(ss-transformed_cam, axis=0)
            std_dev_error = np.std(ss-transformed_cam, axis=0)
            transfer_matrix[2:] += avg_error
            # recalculate based on new data
            transformed_cam = affine_transform(transfer_matrix, cam)
            positional_trace.append(np.hstack((ss, transformed_cam)))
    data = pd.DataFrame(np.vstack(positional_trace), columns=["x_t", "x_l", "x_ct", "x_cl"])

    if is_save:
        data.to_csv("out_of_sample_data_with_p{}_s{}.csv".format(update_period, update_sample))
    return data


def plot_comparison_data(comp_data, smoothing = 1):
    plt.subplot(2, 1, 1)
    plt.plot(running_mean(comp_data.x_t, n=smoothing))
    plt.plot(running_mean(comp_data.x_ct, n=smoothing))
    plt.subplot(2, 1, 2)
    plt.plot(running_mean(comp_data.x_l, n=smoothing))
    plt.plot(running_mean(comp_data.x_cl, n=smoothing))
    plt.show()


def save_fitted_dataframe(comp_data_df, filename):
    """
    saves the data to the filename specified
    :param comp_data_df:
    :param filename:
    :return:
    """
    comp_data_df.to_csv('{}.csv'.format(filename))
    return 0


def compute_rms_error(comp_data):
    return np.mean(np.sqrt(np.square(comp_data.x_t-comp_data.x_ct) + np.square(comp_data.x_l-comp_data.x_cl)))


def make_comparison_of_calibration_periods(transfer_matrix, period_lengths, sample_lengths, oos_filename):
    avg_error = []
    for pl, sl in zip(period_lengths, sample_lengths):
        comp_data = compare_out_of_sample_with_updating(transfer_matrix, filename=oos_filename, is_fit=False,
                                                        update_period=pl, update_sample=sl)
        avg_error.append(compute_rms_error(comp_data))
        print "Average displacement error for n_oos = {}, n_is = {} :: {}".format(pl, sl, compute_rms_error(comp_data))
    return avg_error


if __name__ == "__main__":
    smoothing = 20
    #transfer_matrix, calibration_data_comparison = get_affine_calibration(filename='calibration_2', is_plot=False, is_fit=False)
    transfer_matrix, calibration_data_comparison = get_affine_calibration(filename='calibration_2', is_plot=False, is_fit=True)
    print "Transfer matrix = \n {}".format(transfer_matrix)
    periods = range(10, 210, 10)
    samples = map(lambda x: x/5, periods)
    print periods, samples
    #make_comparison_of_calibration_periods(transfer_matrix, periods, samples, oos_filename='stability_check_2')
    #comp_data = compare_out_of_sample(transfer_matrix, filename='stability_check_2', is_fit=False)
    comp_data = compare_out_of_sample_with_updating(transfer_matrix, filename='stability_check_2', is_fit=False,
                                                    update_period=20, update_sample=4)
    plot_comparison_data(comp_data, smoothing=smoothing)
    print "Average displacement error = {}".format(compute_rms_error(comp_data))

    #print split_into_batches(range(20), 5, 2)

