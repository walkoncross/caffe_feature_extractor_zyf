#!/usr/bin/env python
from load_avg_feats_and_calc_corr import load_avg_feats_and_calc_corr


if __name__ == '__main__':

    avg_feat_fn1 = '../extract_and_save_probs_stats/rlt_probs_stats/corr_prob_feat_avg_for_ids.npy'
    avg_feat_fn2 = '../extract_and_save_probs_stats/rlt_probs_stats/corr_prob_feat_avg_for_ids.npy'

    label1 = 'feat_set1'
    label2 = 'feat_set2'

    save_dir = './rlt_avg_feats_corr'

    load_avg_feats_and_calc_corr(avg_feat_fn1, avg_feat_fn2, label1, label2, save_dir)
