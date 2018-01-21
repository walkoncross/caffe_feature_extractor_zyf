#!/usr/bin/env python
import _init_paths
from load_prob_stats_and_plot_multi_hists import load_prob_stats_and_plot_multi_hists

if __name__ == '__main__':
    # image path: osp.join(image_dir, <each line in image_list_file>)
    stats_fn_list = [
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-webface-corr.txt',
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-webface-corr.txt',
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface-webface.txt',
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface2-webface.txt'
    ]
    prime_idx = 0
    num_ids_list = [10572, 10245, 2564, 8631]
    num_images_list = None

    show_hist = True
    save_dir = './rlt_hist_overlap_hists/corr_webface'

    for i in range(len(stats_fn_list)):
        if i == prime_idx:
            continue

        tmp_stats_fn_list=[stats_fn_list[prime_idx], stats_fn_list[i]]
        tmp_num_ids_list=[num_ids_list[prime_idx], num_ids_list[i]]
        if num_images_list:
            tmp_num_images_list=[num_images_list[prime_idx], [i]]
        else:
            tmp_num_images_list=None

        load_prob_stats_and_plot_multi_hists(tmp_stats_fn_list, tmp_num_ids_list, tmp_num_images_list,
                                         show_hist, save_dir)
