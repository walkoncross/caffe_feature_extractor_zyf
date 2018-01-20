#!/usr/bin/env python
import os
import os.path as osp
import numpy as np

from numpy.linalg import norm


def load_avg_features(feat_fn):
    feat_set = np.load(feat_fn)

    for i in range(feat_set.shape[0]):
        norm_1 = norm(feat_set[i])
        if norm_1 > 0:
            feat_set[i] /= norm_1

    return feat_set


def calc_corr_btw_feat_sets(feat_set1, feat_set2):
    corr_mat12 = np.dot(feat_set1, feat_set2.T)
    corr_mat21 = np.dot(feat_set2, feat_set1.T)

    return corr_mat12, corr_mat21


def analyse_corr_mat_and_save(corr_mat, save_name):
    fp = open(save_name, 'w')

    num_ids = corr_mat.shape[0]

    write_string = 'orig_label max_label  corr[max_label]'
    write_string += '\n'
    fp.write(write_string)

    inter_ids_corr_min = 1.0e10
    inter_ids_corr_avg = 0
    inter_ids_corr_max = -1.0e10

    for i in range(num_ids):
        max_label = np.argmax(corr_mat[i])

        inter_ids_corr_min = min(
            inter_ids_corr_min, corr_mat[i][max_label])
        inter_ids_corr_max = max(
            inter_ids_corr_max, corr_mat[i][max_label])
        inter_ids_corr_avg += corr_mat[i][max_label]

        write_string = '%d    %d    %.4f\n' % (
            i, max_label, corr_mat[i][max_label])

        fp.write(write_string)

    write_string = 'min    ---    %.4f\n' % (
        inter_ids_corr_min)
    fp.write(write_string)

    write_string = 'max    ---    %.4f\n' % (
        inter_ids_corr_max)
    fp.write(write_string)

    write_string = 'avg    ---    %.4f\n' % (
        inter_ids_corr_avg / num_ids)
    fp.write(write_string)

    fp.close()


def load_avg_feats_and_calc_corr(avg_feat_fn1, avg_feat_fn2, label1, label2, save_dir=None):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    feat_set1 = load_avg_features(avg_feat_fn1)
    feat_set2 = load_avg_features(avg_feat_fn2)

    corr_mat12, corr_mat21 = calc_corr_btw_feat_sets(feat_set1, feat_set2)

    fn_prefix12 = osp.join(
        save_dir, 'corr_mat-%s-vs-%s' % (label1, label2))
    corr_mat12_fn = fn_prefix12 + '.npy'
    np.save(corr_mat12_fn, corr_mat12)

    fn_prefix21 = osp.join(
        save_dir, 'corr_mat-%s-vs-%s' % (label2, label1))
    corr_mat21_fn = fn_prefix21 + '.npy'
    np.save(corr_mat21_fn, corr_mat12)

    analyse_corr_mat_and_save(corr_mat12, fn_prefix12 + '.txt')
    analyse_corr_mat_and_save(corr_mat21, fn_prefix21 + '.txt')


if __name__ == '__main__':
    avg_feat_fn1 = '../extract_and_save_probs_stats/rlt_probs_stats/corr_prob_feat_avg_for_ids.npy'
    avg_feat_fn2 = '../extract_and_save_probs_stats/rlt_probs_stats/corr_prob_feat_avg_for_ids.npy'

    label1 = 'feat_set1'
    label2 = 'feat_set2'

    save_dir = './rlt_avg_feats_corr'

    load_avg_feats_and_calc_corr(avg_feat_fn1, avg_feat_fn2, label1, label2, save_dir)
