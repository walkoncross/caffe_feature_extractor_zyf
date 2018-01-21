#!/usr/bin/env python
import os
import os.path as osp
import numpy as np

from numpy.linalg import norm

import _init_paths
from caffe_feature_extractor import CaffeFeatureExtractor


def get_wts(caffe_model_config_json, wt_layer):
    feat_extractor = CaffeFeatureExtractor(caffe_model_config_json)

    wts = feat_extractor.net.params[wt_layer][0].data
    print('wts.shape:', wts.shape)

    return wts


def load_avg_features(feat_fn):
    feat_set = np.load(feat_fn)

    for i in range(feat_set.shape[0]):
        norm_1 = norm(feat_set[i])
        if norm_1 > 0:
            feat_set[i] /= norm_1

    return feat_set


def calc_corr_btw_avg_feats_and_wts(avg_feats, wts):
    corr_mat = np.dot(avg_feats, wts.T)

    return corr_mat


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


def load_avg_feats_and_calc_corr_to_wts(config_json, wt_layer, avg_feat_fn, label, save_dir=None):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    wts = get_wts(config_json, wt_layer)

    feat_set = load_avg_features(avg_feat_fn)

    corr_mat = calc_corr_btw_avg_feats_and_wts(feat_set, wts)

    fn_prefix = osp.join(
        save_dir, 'avg_feats_to_wts_corr_mat-%s' % (label))
    corr_mat_fn = fn_prefix + '.npy'
    np.save(corr_mat_fn, corr_mat)

    analyse_corr_mat_and_save(corr_mat, fn_prefix + '.txt')


if __name__ == '__main__':
    avg_feat_fn = '../extract_and_save_probs_stats/rlt_probs_stats/fc5_feat_avg_for_ids.npy'
    config_json = './extractor_config_sphere64_webface.json'
    wt_layer = 'fc6'

    label = 'feat_set'

    save_dir = './rlt_avg_feats_to_wts_corr'

    load_avg_feats_and_calc_corr_to_wts(
        config_json, wt_layer, avg_feat_fn, label, save_dir)
