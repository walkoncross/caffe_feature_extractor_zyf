#!/usr/bin/env python
from load_probs_and_refine_labels import load_probs_and_refine_labels
import os.path as osp

if __name__ == '__main__':

    prob_thresh = 0.6
    first_new_id = 10572
    max_orig_label = 2

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/corr_prob'
    image_list_file = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'

    # save_dir = None
#    save_dir = osp.join(prob_dir, '..')
    save_dir = './rlt'

    num_images = -1  # <0, means all images
    mirror_input = False

    load_probs_and_refine_labels(prob_dir, prob_thresh,
                                 first_new_id, max_orig_label,
                                 image_list_file, num_images,
                                 save_dir)
