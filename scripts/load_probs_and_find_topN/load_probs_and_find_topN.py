#!/usr/bin/env python
import os
import os.path as osp
import numpy as np


def load_feature(prob_dir, img_fn):
    feat_fn = osp.splitext(img_fn)[0] + '.npy'
    feat_fn = osp.join(prob_dir, feat_fn)

    feat = np.load(feat_fn)

    return feat


def load_probs_and_find_topN(prob_dir, num_ids,
                             image_list_file, num_images=-1,
                             save_dir=None, top_n=10
                             ):
    print 'prob_dir: ', prob_dir
    if not save_dir:
        save_dir = osp.abspath(osp.join(prob_dir, '..'))
    else:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

    output_fn = 'top_%d_per_id_img_list.txt' % top_n

    fp = open(image_list_file, 'r')

    batch_size = 100

    img_list_dict = {}
    prob_list_dict = {}

    ttl_img_cnt = 0

    for line in fp:
        if line.startswith('#'):
            continue

        spl = line.split()
        img_fn = spl[0].strip()
        label = int(spl[1])

        prob = load_feature(prob_dir, img_fn)[label]

        if label not in img_list_dict.keys():
            img_list_dict[label] = [img_fn]
            prob_list_dict[label] = [prob]
        else:
            img_list_dict[label].append(img_fn)
            prob_list_dict[label].append(prob)

        ttl_img_cnt += 1

        if (ttl_img_cnt % batch_size == 0) or (
                num_images > 0 and ttl_img_cnt == num_images):
            print '\n===> Processing %5d images' % (ttl_img_cnt)

        if (num_images > 0 and ttl_img_cnt == num_images):
            break

    fp.close()

    output_fp = open(osp.join(save_dir, output_fn), 'w')

    output_fp.write(
        'image_name        orig_label  prob[orig_label]\n')

    for i in range(num_ids):
        img_list = img_list_dict[i]
        prob_list = prob_list_dict[i]

        sorted_idx = np.argsort(np.array(prob_list))
        cnt = top_n
        if len(img_list) < top_n:
            cnt = len(img_list)

        for j in sorted_idx[-1:-cnt - 1:-1]:
            write_string = '%s  %d  %g\n' % (img_list[j], i, prob_list[j])
            output_fp.write(write_string)

        if (i + 1) % batch_size == 0:
            output_fp.flush()

    output_fp.close()


if __name__ == '__main__':
    prob_dir = '../load_feats_and_stat_corr_to_avg_feats/rlt_stats_feats_corr_to_avg_feats/'
    image_list_file = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'

    num_ids = 3
    num_images = -1
    top_n = 10
    save_dir = './rlt_top_%d_img_list' % top_n

    load_probs_and_find_topN(prob_dir, num_ids,
                             image_list_file, num_images,
                             save_dir, top_n)
