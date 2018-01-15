#!/usr/bin/env python
import os
import os.path as osp
import numpy as np


def load_feature(prob_dir, img_fn):
    feat_fn = osp.splitext(img_fn)[0] + '.npy'
    feat_fn = osp.join(prob_dir, feat_fn)

    feat = np.load(feat_fn)

    return feat


def load_feature_list(prob_dir, img_list):
    probs_list = []
    for fn in img_list:
        feat = load_feature(prob_dir, fn)
        probs_list.append(feat)

    return probs_list


def load_probs_and_calc_stats(prob_dir,
                              probs_len, max_orig_label,
                              image_list_file, num_images=-1,
                              save_dir=None):
    # print 'prob_dir: ', prob_dir
    if not save_dir:
        save_dir = osp.abspath(osp.join(prob_dir, '..'))
    else:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

    num_ids = max_orig_label + 1

    is_train_dataset = (probs_len==num_ids)

    cnt_per_id_vec = np.zeros(num_ids, dtype=np.int32)

    probs_sum_vec = np.zeros((num_ids, probs_len), dtype=np.float32)
    probs_sqsum_vec = np.zeros((num_ids, probs_len), dtype=np.float32)

    fp = open(image_list_file, 'r')

    batch_size = 512

    ttl_img_cnt = 0

    for line in fp:
        if line.startswith('#'):
            continue

        spl = line.split()
        img_fn = spl[0].strip()
        orig_label = int(spl[1])

        ttl_img_cnt += 1

        probs = load_feature(prob_dir, img_fn)

        probs_sum_vec[orig_label] += probs.T
        probs_sqsum_vec[orig_label] += np.square(probs.T)

        cnt_per_id_vec[orig_label] += 1

        if (np.mod(ttl_img_cnt, batch_size) == 0 or
                (num_images > 0 and ttl_img_cnt == num_images)):
            print '\n===> Processed %5d images' % (ttl_img_cnt)

        if (num_images > 0 and ttl_img_cnt == num_images):
            break

    fp.close()

    cnt_per_id_fn = osp.join(save_dir, 'stats-cnt_per_id.npy')
    np.save(cnt_per_id_fn, cnt_per_id_vec)

    output_fn = 'stats-max-label-info.txt'
    output_fp = open(osp.join(save_dir, output_fn), 'w')

    write_string = 'orig_label max_label  probs_avg[max_label]  probs_std[max_label]'
    if is_train_dataset:
        write_string += '  probs_avg[orig_label]  probs_std[orig_label]'
    write_string += '\n'
    output_fp.write(write_string)

    for i in range(num_ids):
        probs_sum_vec[i] /= cnt_per_id_vec[i]
        probs_sqsum_vec[i] /= cnt_per_id_vec[i]
        probs_sqsum_vec[i] -= np.square(probs_sum_vec[i])

        max_label = np.argmax(probs_sum_vec[i])
        write_string = '%d    %d    %.4f    %.4f' % (
            i, max_label, probs_sum_vec[i][max_label], probs_sqsum_vec[i][max_label], )
        write_string += '    %.4f    %.4f' % (
            probs_sum_vec[i][i], probs_sqsum_vec[i][i])
        write_string += '\n'
        output_fp.write(write_string)

    probs_avg_fn = osp.join(save_dir, 'stats-probs_avg_vec.npy')
    probs_std_fn = osp.join(save_dir, 'stats-probs_std_vec.npy')

    np.save(probs_avg_fn, probs_sum_vec)
    np.save(probs_std_fn, probs_sqsum_vec)

    return cnt_per_id_vec, probs_sum_vec, probs_sqsum_vec


if __name__ == '__main__':

    probs_len = 10572
    max_orig_label = 2

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/corr_prob'
    image_list_file = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'

    num_images = -1

    # save_dir = None
    save_dir = osp.join(prob_dir, '..')

    load_probs_and_calc_stats(prob_dir, probs_len,
                              max_orig_label,
                              image_list_file, num_images,
                              save_dir)
