#!/usr/bin/env python
from load_prob_stats_and_refine_labels import load_prob_stats_and_refine_labels
import os.path as osp

if __name__ == '__main__':

    prob_threshs = np.arange(0.5, 0.85, 0.05)
    first_new_id = 10572

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/'
    image_list_file = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'

    num_images = -1 # <0, means all images
    min_objs = 2

    load_probs_and_calc_stats(prob_dir, probs_len,
                              max_orig_label,
                              image_list_file, num_images,
                              save_dir)