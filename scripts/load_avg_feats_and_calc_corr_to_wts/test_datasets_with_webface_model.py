#!/usr/bin/env python
from load_avg_feats_and_calc_corr_to_wts import load_avg_feats_and_calc_corr_to_wts


if __name__ == '__main__':
    config_json = './extractor_config_sphere64_webface_nfs.json'
    wt_layer = 'fc6'

    avg_feat_fn_list = [
        '../prob-results/asian_probs_on_sphere64_webface/fc5_feat_avg_for_ids.npy',
        '../prob-results/webface_probs_on_sphere64_webface/fc5_feat_avg_for_ids.npy',
        '../prob-results/vggface_probs_on_sphere64_webface/fc5_feat_avg_for_ids.npy',
        '../prob-results/vggface2_probs_on_sphere64_webface/fc5_feat_avg_for_ids.npy'
    ]

    label_list = [
        'asian',
        'webface',
        'vggface',
        'vggface2'
    ]

    save_dir = '../prob-results/avg_fc5_to_fc6_wts_corr_webface'

    num_fns = len(avg_feat_fn_list)
    for i in range(num_fns):
        load_avg_feats_and_calc_corr_to_wts(
            config_json, wt_layer,
            avg_feat_fn_list[i],
            label_list[i], save_dir
        )
