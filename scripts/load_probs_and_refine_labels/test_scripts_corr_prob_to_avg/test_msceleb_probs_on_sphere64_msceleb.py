#!/usr/bin/env python
import _init_paths
from load_probs_and_refine_labels import load_probs_and_refine_labels
import os.path as osp
import numpy as np

if __name__ == '__main__':

    prob_threshs = [0.745, 0.755, 0.8]
    first_new_id = 0
    max_orig_label = 78770

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = r'/home/msceleb_probs_on_sphere64_msceleb/fc5_corr_to_avg_fc5'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/MS-Celeb-1M_clean_list_fixed2_78771_ids_5048805_imgs.txt'

    # save_dir = None
    save_dir = '../../prob-results/msceleb_probs_on_sphere64_msceleb/corr_prob_to_avg_threshed_results'

    num_images = -1  # <0, means all images
    mirror_input = False

    for thresh in prob_threshs:
        load_probs_and_refine_labels(prob_dir, thresh,
                                     first_new_id, max_orig_label,
                                     image_list_file, num_images,
                                     save_dir)
