#!/usr/bin/env python
import os
import os.path as osp
import numpy as np

from load_prob_stats import load_prob_stats
from hist import get_hist, plot_hist, calc_otsu_threshold


def get_auto_label(stats_fn):
    base_name = osp.basename(stats_fn)
    spl = osp.splitext(base_name)[0].split('-')

    if 'max-label' in base_name:
        label_eles = spl[-3:]
        if label_eles[0] == 'info':
            label = label_eles[1] + '_on_' + label_eles[2]
        else:
            label = label_eles[0] + '_on_' + label_eles[1]
    elif 'corr_mat' in base_name:
        label = spl[-1]

    return label


def load_prob_stats_and_calc_hist_thresh(stats_fn, num_ids, label1=None, num_images=-1,
                                         stats_fn2=None, num_ids2=None, label2=None, num_images2=-1,
                                         only_after_bin_val=None,
                                         show_hist=True, save_dir=None,
                                         label_list=None):
    print 'file1: ', stats_fn
    print 'file2: ', stats_fn2

    if save_dir is None:
        save_dir = './rlt_hist_output/'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    if not label1:
        label1 = get_auto_label(stats_fn)

    prob_avg_vec, max_label_vec = load_prob_stats(stats_fn, num_ids)
    print 'prob_avg_vec.shape: ', prob_avg_vec.shape
    hist, bins = get_hist(prob_avg_vec)
    thresh = calc_otsu_threshold(hist, bins, only_after_bin_val)

    print 'Otsu_threshold for file1: ', thresh

    # out_prefix = osp.splitext(stats_fn)[0]
    # out_prefix = osp.join(save_dir, osp.basename(out_prefix))
    out_prefix = osp.join(save_dir, label1)

    hist_out_fn = out_prefix + '_thr%g.png' % thresh
    plot_hist(prob_avg_vec, bins, show_hist, hist_out_fn)

    thresh_out_fn = out_prefix + '_thr%g.txt' % thresh
    cnt_gteq_thresh = np.sum(prob_avg_vec >= thresh)
    cnt_lt_thresh = prob_avg_vec.size - cnt_gteq_thresh
    fp = open(thresh_out_fn, 'w')
    write_string = ('thresh: %g\ncnt(>=thresh):%d\ncnt(<thresh):%d' %
                    (thresh, cnt_gteq_thresh, cnt_lt_thresh)
                    )
    fp.write(write_string)
    fp.close()

    if stats_fn2:
        if not label2:
            label2 = get_auto_label(stats_fn2)

        prob_avg_vec2, max_label_vec2 = load_prob_stats(stats_fn2, num_ids2)
        print 'prob_avg_vec2.shape: ', prob_avg_vec2.shape
        hist, bins = get_hist(prob_avg_vec2)
        thresh2 = calc_otsu_threshold(hist, bins, only_after_bin_val)

        print 'Otsu_threshold for file2: ', thresh2

        # out_prefix2 = osp.splitext(stats_fn2)[0]
        # out_prefix2 = osp.join(save_dir, osp.basename(out_prefix2))
        out_prefix2 = osp.join(save_dir, label2)

        hist_out_fn2 = out_prefix2 + '_thr%g.png' % thresh2
        plot_hist(prob_avg_vec2, bins, show_hist, hist_out_fn2)

        thresh_out_fn = out_prefix2 + '_thr%g.txt' % thresh2
        cnt_gteq_thresh = np.sum(prob_avg_vec2 >= thresh2)
        cnt_lt_thresh = prob_avg_vec2.size - cnt_gteq_thresh
        fp = open(thresh_out_fn, 'w')
        write_string = ('thresh: %g\ncnt(>=thresh):%d\ncnt(<thresh):%d' %
                        (thresh2, cnt_gteq_thresh, cnt_lt_thresh)
                        )
        fp.write(write_string)
        fp.close()

        prob_avg_vec3 = np.hstack((prob_avg_vec, prob_avg_vec2))
        print 'prob_avg_vec.shape: ', prob_avg_vec3.shape
        hist, bins = get_hist(prob_avg_vec3)
        thresh3 = calc_otsu_threshold(hist, bins, only_after_bin_val)

        out_prefix3 = out_prefix + '_and_' + osp.basename(out_prefix2)

        hist_out_fn3 = out_prefix3 + '_thr%g.png' % thresh3
        plot_hist(prob_avg_vec3, None, show_hist, hist_out_fn3)

        print 'Otsu_threshold for file1 and file2: ', thresh3

        thresh_out_fn = out_prefix3 + '_thr%g.txt' % thresh3
        cnt_gteq_thresh = np.sum(prob_avg_vec3 >= thresh3)
        cnt_lt_thresh = prob_avg_vec3.size - cnt_gteq_thresh
        fp = open(thresh_out_fn, 'w')
        write_string = ('thresh: %g\ncnt(>=thresh):%d\ncnt(<thresh):%d' %
                        (thresh3, cnt_gteq_thresh, cnt_lt_thresh)
                        )
        fp.write(write_string)
        fp.close()


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

    # only_after_bin_val = None
    # only_after_bin_val = 0.55
    only_after_bin_val = 0.7

    show_hist = True
    save_dir=None

    load_prob_stats_and_calc_hist_thresh(stats_fn, num_ids, num_images,
                                         stats_fn2, num_ids2, num_images2,
                                         only_after_bin_val, show_hist, save_dir)
