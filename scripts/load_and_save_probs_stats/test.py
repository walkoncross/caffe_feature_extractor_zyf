#!/usr/bin/env python
from load_and_save_probs_stats import load_and_save_probs_stats

if __name__ == '__main__':

    feat_root_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels'
    feat_layer_names = 'fc5,corr_prob'
    is_train_set = False
    max_orig_label = 2

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_list_file = r'.\face_chips_list_with_label.txt'

    save_dir = 'rlt_load_save_probs_stats'
    num_images = -1

    load_and_save_probs_stats(feat_root_dir,
                              feat_layer_names,
                              max_orig_label,
                              image_list_file,
                              save_dir, num_images,
                              is_train_set)
