#!/usr/bin/env python
import os
import os.path as osp
import numpy as np

from load_prob_stats import load_prob_stats
from hist import get_hist, plot_multi_hists, calc_otsu_threshold


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


def load_prob_stats_and_plot_multi_hists(stats_fn_list, num_ids_list,
                                         num_images_list=None,
                                         show_hist=True,
                                         save_dir=None,
                                         label_list=None):
    if save_dir is None:
        save_dir = './rlt_hist_output/'
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

#    hist_list = []
#    bins_list = []
    prob_stats_list = []

    if label_list is None:
        label_list = []
        auto_set_label = True
    else:
        auto_set_label = False

    for i, stats_fn in enumerate(stats_fn_list):
        print('===> Loading prob stats from: ' + stats_fn)
        num_ids = -1
        if num_ids_list:
            num_ids = num_ids_list[i]

        print('num_ids: ', num_ids)

        prob_avg_vec, max_label_vec = load_prob_stats(stats_fn, num_ids)
        print('prob_avg_vec.shape: ', prob_avg_vec.shape)
        prob_stats_list.append(prob_avg_vec)

        if auto_set_label:
            # base_name = osp.basename(stats_fn)
            # spl = osp.splitext(base_name)[0].split('-')
            # label_eles = spl[-3:]
            # if label_eles[0] == 'info':
            #     label = label_eles[1] + '_on_' + label_eles[2]
            # else:
            #     label = label_eles[0] + '_on_' + label_eles[1]
            label = get_auto_label(stats_fn)
            label_list.append(label)

#        print('===> Calc hist')
#        hist, bins = get_hist(prob_avg_vec)

#        print('hist: ', hist)
#        print('bins: ', bins)
        # thresh = calc_otsu_threshold(hist, bins, only_after_bin_val)
        # print 'Otsu_threshold for file1: ', thresh
#        hist_list.append(hist)
#        bins_list.append(bins)

    out_prefix = label_list[0]
    if len(label_list)>1:
         out_prefix += '-overlapped-' + label_list[1]
    if len(label_list)>2:
        out_prefix += '-and-others'

    hist_out_fn = osp.join(save_dir, out_prefix + '.png')
    plot_multi_hists(prob_stats_list, None, True, hist_out_fn, label_list)


if __name__ == '__main__':

    # image path: osp.join(image_dir, <each line in image_list_file>)
    stats_fn_list = [
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-msceleb-msceleb.txt',
        r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-webface-msceleb-corr.txt'
        # r'C:\zyf\dnn_models\face_models\face-datasets-merge\stats-max-label-info-asian-msceleb-corr.txt',
        # r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface-msceleb.txt',
        # r'C:\zyf\dnn_models\face_models\face-datasets-merge\corr_prob-stats-max-label-info-vggface2-msceleb.txt'
    ]
    num_ids_list = [78771, 10572, 10245, 2564, 8631]
    num_images_list = None

    show_hist = True
    save_dir = None

    load_prob_stats_and_plot_multi_hists(stats_fn_list, num_ids_list, num_images_list,
                                         show_hist, save_dir)
