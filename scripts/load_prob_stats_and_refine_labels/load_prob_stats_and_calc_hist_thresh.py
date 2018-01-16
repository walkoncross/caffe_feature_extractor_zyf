#!/usr/bin/env python
import os.path as osp
import numpy as np

from load_prob_stats import load_prob_stats
from hist import get_hist, plot_hist, calc_otsu_threshold


def load_prob_stats_and_calc_hist_thresh(stats_fn, num_ids, num_images=-1,
                                         stats_fn2=None, num_ids2=None, num_images2=-1,
                                         only_after_bin_val=None):
    print 'file1: ', stats_fn
    print 'file2: ', stats_fn2

    prob_avg_vec, max_label_vec = load_prob_stats(stats_fn, num_ids)
    print 'prob_avg_vec.shape: ', prob_avg_vec.shape
    hist, bins = get_hist(prob_avg_vec)
    thresh = calc_otsu_threshold(hist, bins, only_after_bin_val)

    print 'Otsu_threshold for file1: ', thresh

    out_prefix = osp.splitext(stats_fn)[0]

    hist_out_fn = out_prefix + '_thr%g.png' % thresh
    plot_hist(prob_avg_vec, bins, True, hist_out_fn)

    if stats_fn2:
        prob_avg_vec2, max_label_vec2 = load_prob_stats(stats_fn2, num_ids2)
        print 'prob_avg_vec2.shape: ', prob_avg_vec2.shape
        hist, bins = get_hist(prob_avg_vec2)
        thresh2 = calc_otsu_threshold(hist, bins, only_after_bin_val)

        print 'Otsu_threshold for file2: ', thresh2

        out_prefix2 = osp.splitext(stats_fn2)[0]

        hist_out_fn2 = out_prefix2 + '_thr%g.png' % thresh2
        plot_hist(prob_avg_vec2, bins, True, hist_out_fn2)

        prob_avg_vec3 = np.hstack((prob_avg_vec, prob_avg_vec2))
        print 'prob_avg_vec.shape: ', prob_avg_vec3.shape
        hist, bins = get_hist(prob_avg_vec3)
        thresh3 = calc_otsu_threshold(hist, bins, only_after_bin_val)

        out_prefix3 = out_prefix + '_and_' + osp.basename(out_prefix2)
        hist_out_fn3 = out_prefix3 + '_thr%g.png' % thresh3
        plot_hist(prob_avg_vec3, None, True, hist_out_fn3)

        print 'Otsu_threshold for file1 and file2: ', thresh3


if __name__ == '__main__':

    # image path: osp.join(image_dir, <each line in image_list_file>)
    # stats_fn = r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-webface-corr.txt'
    # stats_fn = r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-webface-softmax.txt'
    # stats_fn = r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-msceleb-corr.txt'
    stats_fn = r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-msceleb-softmax.txt'
    num_ids = 10572
    num_images = -1

#    only_after_bin_val=False
    # stats_fn2 = r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-webface-corr.txt'
    # stats_fn2 = r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-webface-softmax.txt'
    # stats_fn2 = r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-msceleb-corr.txt'
    stats_fn2 = r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-msceleb-softmax.txt'
    num_ids2 = 10245
    num_images2 = -1

    only_after_bin_val = 0.55

    load_prob_stats_and_calc_hist_thresh(stats_fn, num_ids, num_images,
                                         stats_fn2, num_ids2, num_images2,
                                         only_after_bin_val)
