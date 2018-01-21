#!/usr/bin/env python
from load_avg_feats_and_calc_corr_to_wts import load_avg_feats_and_calc_corr_to_wts


if __name__ == '__main__':

    avg_feat_fn = '../extract_and_save_probs_stats/rlt_probs_stats/fc5_feat_avg_for_ids.npy'
    config_json = './extractor_config_sphere64_webface.json'
    wt_layer = 'fc6'

    label = 'feat_set'

    save_dir = './rlt_avg_feats_to_wts_corr'

    load_avg_feats_and_calc_corr_to_wts(
        config_json, wt_layer, avg_feat_fn, label, save_dir)
