#!/usr/bin/env python
import _init_paths
import os
import os.path as osp
from load_prob_stats_and_calc_hist_thresh import load_prob_stats_and_calc_hist_thresh

if __name__ == '__main__':
    root_dir = r'C:\zyf\dnn_models\face_models\face-datasets-merge\avg_fc5_to_fc6_wts_corr_webface'
    file_list = os.listdir(root_dir)

    stats_fn_list = [osp.join(root_dir, fn) for fn in file_list]
    # num_ids_list = [78771, 10572, 10245, 2564, 8631]

    num_ids_list = []
    for fn in file_list:
        if fn.endswith('corr_mat-asian.txt'):
            num_ids_list.append(10245)
        elif fn.endswith('corr_mat-webface.txt'):
            num_ids_list.append(10572)
        elif fn.endswith('corr_mat-vggface2.txt'):
            num_ids_list.append(8631)
        elif fn.endswith('corr_mat-vggface.txt'):
            num_ids_list.append(2564)
        elif fn.endswith('corr_mat-msceleb.txt'):
            num_ids_list.append(78771)
        else:
            raise Exception('Unkonw dataset')

    num_images_list = None


# #    only_after_bin_val = False
#     only_after_bin_val = 0.55
# #    only_after_bin_val = 0.5
    only_after_bin_val_list = [0, 0.55, 0.5]

    show_hist = False

    num_fns = len(stats_fn_list)

    save_root_dir = './rlt_hist_avg_fc5_to_fc6_wts_corr/feat_hist_on_webface'

    for bin_val in only_after_bin_val_list:
        save_dir = osp.join(save_root_dir, 'hist_png')
        if bin_val > 0:
            save_dir += '_thr_%g' % bin_val

        for i, stats_fn in enumerate(stats_fn_list):
                num_images1 = -1
                if num_ids_list:
                    num_images1 = num_ids_list[i]
                load_prob_stats_and_calc_hist_thresh(stats_fn_list[i], num_ids_list[i], num_images1,
                                                     None, None, None,
                                                     bin_val, show_hist, save_dir)
