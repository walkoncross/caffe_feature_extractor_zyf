#!/usr/bin/env python
import _init_paths
from load_and_save_probs_stats import load_and_save_probs_stats

if __name__ == '__main__':

    feat_root_dir = '/home/webface_probs_on_sphere64_webface'
    feat_layer_names = 'fc5,corr_prob,prob'
    is_train_set = True
    max_orig_label = 10571

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/webface/webface-aligned-list-10572-ids-450833-objs-170503-213839.txt'

    save_dir = '../../prob-results/webface_probs_on_sphere64_webface'
    num_images = -1 # <0, means all images

    load_and_save_probs_stats(feat_root_dir,
                              feat_layer_names,
                              max_orig_label,
                              image_list_file,
                              save_dir, num_images,
                              is_train_set)
