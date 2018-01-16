#!/usr/bin/env python
import os
import os.path as osp
import numpy as np

from load_prob_stats import load_prob_stats


def load_prob_stats_and_refine_labels(stats_dir, prob_threshs,
                                      first_new_id,
                                      image_list_file, num_images=-1,
                                      min_objs=5):

    cnt_per_id_fn = osp.join(stats_dir, 'stats-cnt_per_id.npy')
    cnt_per_id_vec = np.load(cnt_per_id_fn)
    # print 'cnt_per_id_vec: ', cnt_per_id_vec

    num_ids = cnt_per_id_vec.shape[0]

    # probs_avg_fn = osp.join(stats_dir, 'stats-probs_avg_vec.npy')
    # probs_std_fn = osp.join(stats_dir, 'stats-probs_std_vec.npy')

    # probs_avg_vec = np.load(probs_avg_fn)
    # probs_std_vec = np.save(probs_std_fn)
    stats_fn = osp.join(stats_dir, 'stats-max-label-info.txt')
    prob_avg_vec, max_label_vec = load_prob_stats(stats_dir, num_ids)

    for prob_thresh in prob_threshs:
        new_id_map_mat = np.ones(num_ids, dtype=np.int32) * -1
        new_id_cnt = first_new_id

        for i in range(num_ids):
            if int(cnt_per_id_vec[i]) < min_objs:
                continue

            if prob_avg_vec[i] < prob_thresh:
                new_id_map_mat[i] = new_id_cnt
                new_id_cnt += 1
            else:
                new_id_map_mat[i] = max_label_vec[i]

        new_id_map_fn = 'stats-prob_thresh_%g-nonoverlap-new_id_map.npy' % prob_thresh
        new_id_map_fn = osp.join(stats_dir, new_id_map_fn)
        np.save(new_id_map_fn, new_id_map_mat)

        # test extract_features_for_image_list()
        output_fn1 = 'stats-prob_thresh_%g-nonoverlap-img_list.txt' % prob_thresh
        output_fp1 = open(osp.join(stats_dir, output_fn1), 'w')
        output_fn2 = 'stats-prob_thresh_%g-overlap-img_list.txt' % prob_thresh
        output_fp2 = open(osp.join(stats_dir, output_fn2), 'w')
        output_fn3 = 'stats-prob_thresh-removed-img_list.txt' % prob_thresh
        output_fp3 = open(osp.join(stats_dir, output_fn3), 'w')

        fp = open(image_list_file, 'r')

        batch_size = 512

        ttl_img_cnt = 0

        for line in fp:
            if line.startswith('#'):
                continue

            spl = line.split()
            img_fn = spl[0].strip()
            label = int(spl[1])

            ttl_img_cnt += 1

            if new_id_map_mat[label] < 0:
                write_string = '%s  %d\n' % (img_fn, label)
                output_fp3.write(write_string)
            elif new_id_map_mat[label] < first_new_id:
                write_string = '%s  %d\n' % (img_fn, new_id_map_mat[label])
                output_fp2.write(write_string)
            else:
                write_string = '%s  %d\n' % (img_fn, new_id_map_mat[label])
                output_fp1.write(write_string)

            if np.mod(ttl_img_cnt, batch_size) == 0 or (num_images > 0 and ttl_img_cnt == num_images):
                print '\n===> Processing %5d images' % (ttl_img_cnt)

                output_fp1.flush()
                output_fp2.flush()

            if (num_images > 0 and ttl_img_cnt == num_images):
                break

        fp.close()

        output_fp1.close()
        output_fp2.close()


if __name__ == '__main__':

    prob_threshs = np.arange(0.5, 0.85, 0.05)
    first_new_id = 10572

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/'
    image_list_file = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'

    num_images = -1
    min_objs = 2

    # stats_dir = None
    stats_dir = osp.join(prob_dir, '..')

    load_prob_stats_and_refine_labels(prob_dir, prob_threshs,
                                      first_new_id,
                                      image_list_file, num_images,
                                      min_objs)
