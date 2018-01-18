#!/usr/bin/env python

from load_prob_stats_and_calc_hist_thresh import load_prob_stats_and_calc_hist_thresh

if __name__ == '__main__':
    stats_fn1_list = [r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-webface-corr.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-webface-softmax.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-msceleb-corr.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-msceleb-softmax.txt'
                      ]
    num_ids1 = 10572
    num_images1 = -1

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

    stats_fn4_list = [r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface2-webface.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\softmax_prob-stats-max-label-info-vggface2-webface.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface2-msceleb.txt',
                      r'C:\zyf\dnn_models\face_models\face-datasets-merge\softmax_prob-stats-max-label-info-vggface2-msceleb.txt'
                      ]
    num_ids4 = 2564
    num_images4 = -1

#    only_after_bin_val = False
    only_after_bin_val = 0.55
#    reset_only_after_bin_val = False
    reset_only_after_bin_val = True

    show_hist = False

    for i, stats_fn1 in enumerate(stats_fn1_list):
        if i >= len(stats_fn2_list):
            stats_fn2 = None
        else:
            stats_fn2 = stats_fn2_list[i]

        if reset_only_after_bin_val:
            if 'corr' in stats_fn1:
                only_after_bin_val = 0.5
            else:
                only_after_bin_val = 0.7

        load_prob_stats_and_calc_hist_thresh(stats_fn1, num_ids1, num_images1,
                                             stats_fn2, num_ids2, num_images2,
                                             only_after_bin_val, show_hist)

    for i, stats_fn1 in enumerate(stats_fn1_list):
        if i >= len(stats_fn3_list):
            stats_fn3 = None
        else:
            stats_fn3 = stats_fn3_list[i]

        if reset_only_after_bin_val:
            if 'corr' in stats_fn1:
                only_after_bin_val = 0.5
            else:
                only_after_bin_val = 0.7

        load_prob_stats_and_calc_hist_thresh(stats_fn1, num_ids1, num_images1,
                                             stats_fn3, num_ids3, num_images3,
                                             only_after_bin_val, show_hist)

    for i, stats_fn2 in enumerate(stats_fn2_list):
        if i >= len(stats_fn3_list):
            stats_fn3 = None
        else:
            stats_fn3 = stats_fn3_list[i]

        if reset_only_after_bin_val:
            if 'corr' in stats_fn2:
                only_after_bin_val = 0.5
            else:
                only_after_bin_val = 0.7

        load_prob_stats_and_calc_hist_thresh(stats_fn2, num_ids2, num_images2,
                                             stats_fn3, num_ids3, num_images3,
                                             only_after_bin_val, show_hist)

    for i, stats_fn1 in enumerate(stats_fn1_list):
        if i >= len(stats_fn4_list):
            stats_fn4 = None
        else:
            stats_fn4 = stats_fn4_list[i]

        if reset_only_after_bin_val:
            if 'corr' in stats_fn1:
                only_after_bin_val = 0.5
            else:
                only_after_bin_val = 0.7

        load_prob_stats_and_calc_hist_thresh(stats_fn1, num_ids1, num_images1,
                                             stats_fn4, num_ids4, num_images4,
                                             only_after_bin_val, show_hist)

    for i, stats_fn2 in enumerate(stats_fn2_list):
        if i >= len(stats_fn4_list):
            stats_fn4 = None
        else:
            stats_fn4 = stats_fn4_list[i]

        if reset_only_after_bin_val:
            if 'corr' in stats_fn2:
                only_after_bin_val = 0.5
            else:
                only_after_bin_val = 0.7

        load_prob_stats_and_calc_hist_thresh(stats_fn2, num_ids2, num_images2,
                                             stats_fn4, num_ids4, num_images4,
                                             only_after_bin_val, show_hist)

    for i, stats_fn3 in enumerate(stats_fn3_list):
        if i >= len(stats_fn4_list):
            stats_fn4 = None
        else:
            stats_fn4 = stats_fn4_list[i]

        if reset_only_after_bin_val:
            if 'corr' in stats_fn3:
                only_after_bin_val = 0.5
            else:
                only_after_bin_val = 0.7

        load_prob_stats_and_calc_hist_thresh(stats_fn3, num_ids3, num_images3,
                                             stats_fn4, num_ids4, num_images4,
                                             only_after_bin_val, show_hist)
