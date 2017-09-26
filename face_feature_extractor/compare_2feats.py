#! /usr/bin/env python

import os
import sys
import os.path as osp

import numpy as np
from numpy.linalg import norm


def load_npy(npy_file):
    mat = None
    if osp.exists(npy_file):
        mat = np.load(npy_file)
    else:
        print 'Can not find file: ', npy_file

    return mat


def calc_similarity(feat1, feat2):
    feat1_norm = norm(feat1)
    feat2_norm = norm(feat2)

    sim = np.dot(feat1, feat2) / (feat1_norm * feat2_norm)

    return sim


def compare_feats(file1, file2):
    print 'Load feat file 1: ', file1
    ft1 = load_npy(file1)
    if ft1 is None:
        print "Failed to load feature1's .npy"
        return None

    print 'Load feat file 2: ', file2
    ft2 = load_npy(file2)
    if ft2 is None:
        print "Failed to load feature2's .npy"
        return None

    sim = calc_similarity(ft1, ft2)
    print '--->similarity: ', sim

    return sim


if __name__ == '__main__':
    if len(sys.argv) >= 2:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
    else:
        file1 = 'results/0_a.jpg_0_rect[534_274_874_710].npy'
        file2 = 'results/0_b.jpg_0_rect[493_321_1040_1024].npy'

    compare_feats(file1, file2)
