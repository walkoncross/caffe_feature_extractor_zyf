#!/usr/bin/env python
import _init_paths
from load_probs_and_refine_labels import load_probs_and_refine_labels
import os.path as osp
import numpy as np

if __name__ == '__main__':

    prob_threshs = np.arange(0.4, 0.85, 0.05)
    first_new_id = 78771
    max_orig_label = 10571


    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = r'/home/webface_probs_on_sphere64_msceleb/corr_prob'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/webface/webface-aligned-list-10572-ids-450833-objs-170503-213839.txt'

    # save_dir = None
    save_dir = '../webface_probs_on_sphere64_msceleb/corr_prob_threshed_results'

    num_images = -1  # <0, means all images
    mirror_input = False

    for thresh in prob_threshs:
        load_probs_and_refine_labels(prob_dir, thresh,
                                     first_new_id, max_orig_label,
                                     image_list_file, num_images,
                                     save_dir)
