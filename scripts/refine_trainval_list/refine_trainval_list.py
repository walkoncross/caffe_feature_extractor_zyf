# -*- coding: utf-8 -*-
"""
Created on Sat Jan 27 06:27:01 2018

@author: zhaoy
"""

import os
import os.path as osp


def load_refined_list_dict(fn):
    fp = open(fn, 'r')
    # skip the first line
    line = fp.readline()
    img_list_dict = {}
    for line in fp:
        spl = line.split()

        img_fn = spl[0]
#        old_label = spl[1]
        new_label = spl[2]

        img_list_dict[img_fn] = new_label

    fp.close()

    return img_list_dict


def refine_trainval_list(trainval_list_fns,
                         refined_list_fn,
                         refined_nonoverlap_list_fn=None,
                         save_suffix=None):

    if not save_suffix:
        save_suffix = '_refined'

    refined_list = load_refined_list_dict(refined_list_fn)
    refined_nonoverlap_list = None
    if refined_nonoverlap_list_fn:
        refined_nonoverlap_list = load_refined_list_dict(
            refined_nonoverlap_list_fn)

    for fn in trainval_list_fns:
        fp = open(fn, 'r')
        spl = osp.splitext(fn)
        refined_list_fn = spl[0] + save_suffix + spl[1]
        refined_list_fn2 = spl[0] + save_suffix + '-nonoverlap' + spl[1]

        print 'refined image list will save into: ', refined_list_fn
        print 'nonoverlapped image list will save into: ', refined_list_fn2
        fp_out = open(refined_list_fn, 'w')
        fp_out2 = open(refined_list_fn2, 'w')

        for line in fp:
            spl = line.split()

            label = refined_list.get(spl[0], -1)

            if label < 0:
                if refined_nonoverlap_list_fn:
                    label = refined_nonoverlap_list.get(spl[0], spl[1])
                fp_out2.write(spl[0] + ' ' + label + '\n')
            else:
                fp_out.write(spl[0] + ' ' + label + '\n')

        fp_out.close()
        fp_out2.close()


if __name__ == '__main__':
    # trainval_list_fns = [
    #     '/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/msceleb_fixed2_train_list_ratio-0.95_78771-ids_4833609-objs_171208-230236.txt',
    #     '/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/msceleb_fixed2_val_list_ratio-0.05_70111-ids_215196-objs_171208-230236.txt'
    # ]
    trainval_list_fns = [
        '/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/msceleb_fixed2_train_list_ratio-0.9_78771-ids_4579332-objs_171208-230319.txt',
        '/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/msceleb_fixed2_val_list_ratio-0.1_75820-ids_469473-objs_171208-230319.txt',
        '/disk2/zhaoyafei/face-recog-train/train-val-lists/msceleb-1m/MS-Celeb-1M_clean_list_fixed2_78771_ids_5048805_imgs.txt'
    ]    
    refined_list_fn = '../prob-results/msceleb_probs_on_sphere64_msceleb/corr_prob_to_avg_threshed_results/prob_thresh_0.5333-overlap-img_list.txt'
    refined_nonoverlap_list_fn = '../prob-results/msceleb_probs_on_sphere64_msceleb/corr_prob_to_avg_threshed_results/prob_thresh_0.5333-nonoverlap-img_list.txt'

    save_suffix = '_refined_thr0.5333'
    refine_trainval_list(trainval_list_fns, refined_list_fn,
                         refined_nonoverlap_list_fn, save_suffix)
