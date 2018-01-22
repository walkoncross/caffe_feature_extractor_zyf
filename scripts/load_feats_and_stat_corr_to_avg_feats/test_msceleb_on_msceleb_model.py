#!/usr/bin/env python
from load_feats_and_stat_corr_to_avg_feats import load_feats_and_stat_corr_to_avg_feats


if __name__ == '__main__':
#    image_dir = r"/disk2/data/FACE/celeb-1m-mtcnn-aligned/msceleb_align/Faces-Aligned/"
    img_list_fn = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/MS-Celeb-1M_clean_list_fixed2_78771_ids_5048805_imgs.txt'
    feat_dir = '/home/msceleb_probs_on_sphere64_msceleb/fc5'
    avg_feat_fn = '../prob-results/msceleb_probs_on_sphere64_msceleb/fc5_feat_avg_for_ids.npy'
    num_images = -1
    save_dir = '/home/msceleb_probs_on_sphere64_msceleb/fc5_corr_to_avg_fc5'

    load_feats_and_stat_corr_to_avg_feats(img_list_fn, feat_dir,
                                 avg_feat_fn, save_dir, num_images)