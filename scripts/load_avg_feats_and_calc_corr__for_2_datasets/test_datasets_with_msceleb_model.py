#!/usr/bin/env python
from load_avg_feats_and_calc_corr import load_avg_feats_and_calc_corr


if __name__ == '__main__':

    avg_feat_fn_list = [
        '../prob-results/asian_probs_on_sphere64_msceleb/fc5_feat_avg_for_ids.npy',
        '../prob-results/webface_probs_on_sphere64_msceleb/fc5_feat_avg_for_ids.npy',
        '../prob-results/vggface_probs_on_sphere64_msceleb/fc5_feat_avg_for_ids.npy',
        '../prob-results/vggface2_probs_on_sphere64_msceleb/fc5_feat_avg_for_ids.npy',
        '../prob-results/msceleb_probs_on_sphere64_msceleb/fc5_feat_avg_for_ids.npy'
    ]

    label_list = [
        'asian',
        'webface',
        'vggface',
        'vggface2',
        'msceleb'
    ]

    save_dir = '../prob-results/inter_datasets_avg_fc5_feats_corr'

    num_fns = len(avg_feat_fn_list)
    for i in range(num_fns):
        load_avg_feats_and_calc_corr(
            avg_feat_fn_list[i], avg_feat_fn_list[i],
            label_list[i], label_list[i], save_dir
        )

        for j in range(i + 1, num_fns):
            load_avg_feats_and_calc_corr(
                avg_feat_fn_list[i], avg_feat_fn_list[j],
                label_list[i], label_list[j], save_dir
            )
