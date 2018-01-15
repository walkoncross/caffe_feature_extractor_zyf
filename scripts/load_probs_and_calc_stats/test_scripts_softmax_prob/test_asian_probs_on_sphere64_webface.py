#!/usr/bin/env python
import _init_paths
from load_probs_and_calc_stats import load_probs_and_calc_stats
import os.path as osp


if __name__ == '__main__':

    probs_len = 10572
    max_orig_label = 10244


    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = r'/home/asian_probs_on_sphere64_webface/prob'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/asian/face_asian_train_list_noval_10245-ids_540735-objs_170818-225846-norootdir.txt'

    # save_dir = None
    save_dir = '../asian_probs_on_sphere64_webface/softmax_prob_stats'

    num_images = -1  # <0, means all images

    load_probs_and_calc_stats(prob_dir, probs_len,
                              max_orig_label,
                              image_list_file, num_images,
                              save_dir)