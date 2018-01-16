#!/usr/bin/env python
import _init_paths
from load_prob_stats_and_refine_labels import load_prob_stats_and_refine_labels
import os.path as osp


if __name__ == '__main__':

    prob_threshs = np.arange(0.5, 0.85, 0.05)
    first_new_id = 78771

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = r'../../prob-results/webface_probs_on_sphere64_msceleb/softmax_prob_stats'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/webface/webface-aligned-list-10572-ids-450833-objs-170503-213839.txt'

    num_images = -1  # <0, means all images
    min_objs = 5

    load_probs_and_calc_stats(prob_dir, probs_len,
                              max_orig_label,
                              image_list_file, num_images,
                              min_objs)