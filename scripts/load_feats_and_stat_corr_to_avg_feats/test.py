#!/usr/bin/env python
from load_feats_and_stat_corr_to_avg_feats import load_feats_and_stat_corr_to_avg_feats


if __name__ == '__main__':

    img_list_fn = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'
    feat_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/fc5/'
    avg_feat_fn = '../extract_and_save_probs_stats/rlt_probs_stats/fc5_feat_avg_for_ids.npy'
    num_images = -1
    save_dir = './rlt_stats_feats_corr_to_avg_feats'

    load_feats_and_stat_corr_to_avg_feats(img_list_fn, feat_dir,
                                 avg_feat_fn, save_dir, num_images)
