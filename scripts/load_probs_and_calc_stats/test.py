#!/usr/bin/env python
from load_probs_and_calc_stats import load_probs_and_calc_stats
import os.path as osp

if __name__ == '__main__':

    probs_len = 10572
    max_orig_label = 2

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/corr_prob'
    image_list_file = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'

    num_images = -1

    # save_dir = None
    save_dir = osp.join(prob_dir, '..')

    load_probs_and_calc_stats(prob_dir, probs_len,
                              max_orig_label,
                              image_list_file, num_images,
                              save_dir)
