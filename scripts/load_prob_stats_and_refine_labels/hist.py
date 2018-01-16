#!/usr/bin/env python
import numpy as np

import matplotlib
import psutil

if psutil.LINUX:
    matplotlib.use('agg')

import matplotlib.pyplot as plt


def calc_otsu_threshold(hist, bins, only_after_bin_val=False):
    #    hist = np.array(hist)
    total = hist.sum()
    print 'hist.sum(): ', total
    bin_num = len(hist)

    bins_means = (bins[0:-1] + bins[1:]) * 0.5
    print 'bins_means: ', bins_means

    sumB = 0.0
    wB = 0.0
    maximum = 0.0

    sum1 = np.dot(bins_means, hist.T)

#    mean = sum1 / total
#    print mean
    for i in range(bin_num):
        wB = wB + hist[i]
        wF = total - wB
        if wB == 0 or wF == 0:
            continue

        sumB = sumB + bins_means[i] * hist[i]

        if only_after_bin_val > 0 and bins_means[i] < only_after_bin_val:
            continue

        mF = (sum1 - sumB) / wF
        between = wB * wF * ((sumB / wB) - mF) * ((sumB / wB) - mF)

        if between >= maximum:
            thresh = bins_means[i]
            maximum = between

    return thresh


def get_hist(arr, bins=None):
    if not bins:
        bins = np.linspace(0, 1, 101, endpoint=True)

    hist, bins = np.histogram(arr, bins)

#    print 'hist:', hist
#    print 'bins:', bins
#
#    print 'hist.shape:', hist.shape

    return hist, bins


def plot_hist(arr, bins=None, show=True, save_dir=None):

    if bins is None:
        bins = np.linspace(0, 1, 100, endpoint=True)

    plt.hist(arr, bins=bins)  # arguments are passed to np.histogram
    plt.title("Histogram")
    if save_dir:
        plt.savefig(save_dir, bbox_inches='tight')
    if show:
        plt.show()
