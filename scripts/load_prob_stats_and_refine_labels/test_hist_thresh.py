#!/usr/bin/env python

from load_prob_stats_and_calc_hist_thresh import load_prob_stats_and_calc_hist_thresh

if __name__ == '__main__':
    stats_fn_list = [r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-webface-corr.txt',
                     r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-webface-softmax.txt',
                     r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-msceleb-corr.txt',
                     r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-msceleb-softmax.txt'
                     ]
    num_ids = 10572
    num_images = -1

    stats_fn2_list = [r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-webface-corr.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-webface-softmax.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-msceleb-corr.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-msceleb-softmax.txt'
                      ]
    num_ids2 = 10245
    num_images2 = -1

    stats_fn3_list = [r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface-webface.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\softmax_prob-stats-max-label-info-vggface-webface.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface-msceleb.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\softmax_prob-stats-max-label-info-vggface-msceleb.txt'
                      ]
    num_ids3 = 2564
    num_images3 = -1

#    only_after_bin_val = 0.55
    only_after_bin_val = False

    for i, stats_fn in enumerate(stats_fn_list):
        if i >= len(stats_fn2_list):
            stats_fn2 = None
        else:
            stats_fn2 = stats_fn2_list[i]

        load_prob_stats_and_calc_hist_thresh(stats_fn, num_ids, num_images,
                                             stats_fn2, num_ids2, num_images2,
                                             only_after_bin_val)

    for i, stats_fn in enumerate(stats_fn_list):
        if i >= len(stats_fn3_list):
            stats_fn3 = None
        else:
            stats_fn3 = stats_fn3_list[i]

        load_prob_stats_and_calc_hist_thresh(stats_fn, num_ids, num_images,
                                             stats_fn3, num_ids3, num_images3,
                                             only_after_bin_val)

    for i, stats_fn2 in enumerate(stats_fn2_list):
        if i >= len(stats_fn3_list):
            stats_fn3 = None
        else:
            stats_fn3 = stats_fn3_list[i]

        load_prob_stats_and_calc_hist_thresh(stats_fn2, num_ids2, num_images2,
                                             stats_fn3, num_ids3, num_images3,
                                             only_after_bin_val)
