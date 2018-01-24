#!/usr/bin/env python
from load_probs_and_find_topN import load_probs_and_find_topN

if __name__ == '__main__':

    prob_dir = '/home/msceleb_probs_on_sphere64_msceleb/fc5_corr_to_avg_fc5'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/MS-Celeb-1M_clean_list_fixed2_78771_ids_5048805_imgs.txt'

    num_ids = 78771
    num_images = -1
    top_n = 10
    save_dir = '../prob-results/msceleb_probs_on_sphere64_msceleb/rlt_top_%d_img_list' % top_n

    load_probs_and_find_topN(prob_dir, num_ids,
                             image_list_file, num_images,
                             save_dir, top_n)
