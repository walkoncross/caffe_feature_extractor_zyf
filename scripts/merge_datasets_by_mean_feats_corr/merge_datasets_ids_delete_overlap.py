#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 06:52:34 2018

@author: zhaoy
"""
from __future__ import print_function

import os
import os.path as osp
import numpy as np


def get_inter_dataset_corr_fn(label1, label2):
    fn = 'corr_mat-%s-vs-%s.txt' % (label1, label2)

    return fn


def get_output_fn_prefix(data_labels, i, j):
    if i <= j:
        raise Exception('Must have i>j in get_output_fn()')

    fn_prefix = data_labels[i] + '_merge_with'

    for k in range(j, i):
        fn_prefix += '_' + data_labels[k]

    return fn_prefix


def load_inter_dataset_corr_file(fn):
    comments = ['#', 'm', 'a']
    id_map_mat = np.loadtxt(fn, skiprows=1, comments=comments)

    return id_map_mat


def merge_datasets_ids_without_overlap(data_labels,
                                       inter_datasets_corr_file_dir,
                                       thresh,
                                       init_num_ids,
                                       save_dir=None):
    if save_dir is None:
        save_dir = inter_datasets_corr_file_dir

    elif not osp.exists(save_dir):
        os.makedirs(save_dir)

    num_new_ids = init_num_ids
    num_datasets = len(data_labels)

    fn_log = osp.join(save_dir, 'process-log.txt')
    fp_log = open(fn_log, 'w')

    for i in range(1, num_datasets):
        remain_ids = None
        write_string = '\n===> Merging dataset: {}\n'.format(data_labels[i])
        print(write_string)
        fp_log.write(write_string)

        for j in range(0, i):
            fn = get_inter_dataset_corr_fn(data_labels[i], data_labels[j])
            fn = osp.join(inter_datasets_corr_file_dir, fn)
            write_string = '---> Load data from: {}\n'.format(fn)
            print(write_string)
            fp_log.write(write_string)

            id_map_mat = load_inter_dataset_corr_file(fn)
            tmp_remain_ids = id_map_mat[:, 2] <= thresh

            num_orig_ids = len(tmp_remain_ids)
            num_remain_ids = tmp_remain_ids.sum()

            write_string = 'num of orig ids: {}\n'.format(num_orig_ids)
            write_string += 'ids overlapped with {}: {}\n'.format(
                data_labels[j], num_orig_ids - num_remain_ids)
            write_string += 'remained ids: {}\n'.format(num_remain_ids)
            print(write_string)
            fp_log.write(write_string)

            if remain_ids is None:
                remain_ids = tmp_remain_ids
            else:
                remain_ids = np.logical_and(remain_ids, tmp_remain_ids)

            num_remain_ids = remain_ids.sum()
            write_string = '-----------------\n'
            write_string += 'ids overlapped with all the previous datasets: {}\n'.format(
                num_orig_ids - num_remain_ids)
            write_string += 'remained ids: {}\n'.format(num_remain_ids)
            print(write_string)
            fp_log.write(write_string)

        fn_prefix = get_output_fn_prefix(data_labels, i, 0)
        fn_out = osp.join(save_dir, fn_prefix + '_new_id_map.txt')
        fp_out = open(fn_out, 'w')

        write_string = '---> save new_id_map into file: {}\n'.format(fn_out)
        print(write_string)
        fp_log.write(write_string)

        fp_out.write('orig_id  new_id\n')
        new_id = num_new_ids
        for i, flag in enumerate(remain_ids):
            if flag:
                fp_out.write('%d  %d\n' % (i, new_id))
                new_id += 1
            else:
                fp_out.write('%d  %d\n' % (i, -1))
        fp_out.close()

        num_new_ids = new_id

        write_string = '---> total new ids added: {}\n'.format(num_new_ids - init_num_ids)
        print(write_string)
        fp_log.write(write_string)

    fp_log.close()


if __name__ == '__main__':
    data_labels = [
        'msceleb',
        'asian',
        'webface',
        'vggface2',
        'vggface'
    ]

    inter_datasets_corr_file_dir = r'C:\zyf\dnn_models\face_models\face-datasets-merge\inter_datasets_avg_fc5_feats_corr_msceleb'
    thresh = 0.8
    init_num_ids = 78771
    save_dir = './rlt_merge_id_map_delete_overlap'

    merge_datasets_ids_without_overlap(data_labels,
                                       inter_datasets_corr_file_dir,
                                       thresh,
                                       init_num_ids,
                                       save_dir)
