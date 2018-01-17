#!/usr/bin/env python
import _init_paths
from load_and_save_probs_stats import load_and_save_probs_stats

if __name__ == '__main__':

    feat_root_dir = '/home/msceleb_probs_on_sphere64_msceleb'
    feat_layer_names = 'fc5,corr_prob,prob'
    is_train_set = True
    max_orig_label = 78770

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r"/disk2/data/FACE/celeb-1m-mtcnn-aligned/msceleb_align/Faces-Aligned/"
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/MS-Celeb-1M_clean_list_fixed2_78771_ids_5048805_imgs.txt'

    save_dir = '../../prob-results/msceleb_probs_on_sphere64_msceleb'
    num_images = -1 # <0, means all images

    load_and_save_probs_stats(feat_root_dir,
                              feat_layer_names,
                              max_orig_label,
                              image_list_file,
                              save_dir, num_images,
                              is_train_set)
