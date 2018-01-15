#!/usr/bin/env python
from load_probs_and_refine_labels import load_probs_and_refine_labels
import os.path as osp
import numpy as np

if __name__ == '__main__':

    prob_threshs = np.arange(0.3, 0.85, 0.05)
    first_new_id = 10572
    max_orig_label = 2

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/corr_prob'
    image_list_file = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'

    # save_dir = None
    save_dir = osp.join(prob_dir, '..')

    num_images = -1  # <0, means all images
    mirror_input = False

    for thresh in prob_threshs:
        load_probs_and_refine_labels(prob_dir, thresh,
                                     first_new_id, max_orig_label,
                                     image_list_file, num_images,
                                     save_dir)
