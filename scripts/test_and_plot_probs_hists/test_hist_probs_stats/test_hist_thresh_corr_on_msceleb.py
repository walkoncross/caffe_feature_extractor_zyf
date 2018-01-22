#!/usr/bin/env python
import _init_paths
import os.path as osp
from load_prob_stats_and_calc_hist_thresh import load_prob_stats_and_calc_hist_thresh

if __name__ == '__main__':
    stats_fn_list = [
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-msceleb-msceleb.txt',
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-msceleb-corr.txt',
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-msceleb-corr.txt',
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface-msceleb.txt',
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface2-msceleb.txt'
    ]
    num_ids_list = [78771, 10572, 10245, 2564, 8631]
    num_images_list = None


# #    only_after_bin_val = False
#     only_after_bin_val = 0.55
# #    only_after_bin_val = 0.5
    only_after_bin_val_list = [0, 0.55, 0.5]

    show_hist = False

    num_fns = len(stats_fn_list)

    save_root_dir = './rlt_hist_probs_stats/corr_msceleb'

    for bin_val in only_after_bin_val_list:
        save_dir = osp.join(save_root_dir, 'hist_png')
        if bin_val > 0:
            save_dir += '_thr_%g' % bin_val

        for i, stats_fn in enumerate(stats_fn_list):
            for j in range(i + 1, num_fns):
                num_images1 = -1
                num_images2 = -1
                if num_ids_list:
                    num_images1 = num_ids_list[i]
                    num_images2 = num_ids_list[i]
                load_prob_stats_and_calc_hist_thresh(stats_fn_list[i], num_ids_list[i], None, num_images1,
                                                     stats_fn_list[j], num_ids_list[j], None, num_images2,
                                                     bin_val, show_hist, save_dir)
