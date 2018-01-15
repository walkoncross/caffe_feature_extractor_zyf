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


def process_image_list(prob_dir, prob_thresh,
                       last_new_id, new_id_map,
                       calc_orig_label_prob,
                       out_fp1, out_fp2,
                       img_list, label_list=None
                       ):
    print 'prob_dir: ', prob_dir

    for i, img_fn in enumerate(img_list):
        probs = load_feature(prob_dir, img_fn)

        # print 'probs.shape:', probs.shape

        # if label_list:
        #     print 'original label: ', label_list[i]
        #     if calc_orig_label_prob:
        #         print 'prob[orig_label]: ', probs[label_list[i]]

        max_label = np.argmax(probs)
        # print 'max_label=%5d, probs[max_label]=%.4f' % (max_label,
        # probs[max_label])

        new_label = -1

        if probs[max_label] >= prob_thresh:
            new_label = max_label
        elif label_list:
            if new_id_map[label_list[i]] < 0:
                last_new_id += 1
                new_id_map[label_list[i]] = last_new_id

            new_label = new_id_map[label_list[i]]

        write_string = "%s\t%5d\t%5d\t%.4f" % (
            img_fn, new_label, max_label, probs[max_label])

        if label_list:
            write_string += "\t%5d" % (label_list[i])
            if calc_orig_label_prob:
                write_string += "\t%.4f" % (probs[label_list[i]])

        write_string += '\n'

        if probs[max_label] < prob_thresh:
            out_fp1.write(write_string)
        else:
            out_fp2.write(write_string)

    return last_new_id


def load_probs_and_refine_labels(prob_dir, prob_thresh,
                                    first_new_id, max_orig_label,
                                    image_list_file, num_images=-1,
                                    save_dir=None
                                    ):
    print 'prob_dir: ', prob_dir
    if not save_dir:
        save_dir = osp.abspath(osp.join(prob_dir, '..'))
    else:
        if not osp.exists(save_dir):
            os.makedirs(save_dir)

    new_id_map = np.ones(max_orig_label + 1, dtype=np.int32) * (-1)

    calc_orig_label_prob = (first_new_id == 0)

    # test extract_features_for_image_list()
    output_fn1 = 'prob_thresh_%g-nonoverlap-img_list.txt' % prob_thresh
    output_fp1 = open(osp.join(save_dir, output_fn1), 'w')
    output_fn2 = 'prob_thresh_%g-overlap-img_list.txt' % prob_thresh
    output_fp2 = open(osp.join(save_dir, output_fn2), 'w')

    output_fp1.write(
        'image_name        new_label  max_label  prob[max_label]  orig_label  prob[orig_label]\n')
    output_fp2.write(
        'image_name        new_label  max_label  prob[max_label]  orig_label  prob[orig_label]\n')

    fp = open(image_list_file, 'r')

    batch_size = 512

    img_list = []
    label_list = []

    ttl_img_cnt = 0
    batch_img_cnt = 0
    batch_cnt = 0

    last_new_id = first_new_id - 1

    for line in fp:
        if line.startswith('#'):
            continue

        spl = line.split()
        img_list.append(spl[0].strip())

        if (len(spl) > 1):
            label_list.append(int(spl[1]))

        batch_img_cnt += 1
        ttl_img_cnt += 1

        if batch_img_cnt == batch_size or (num_images > 0 and ttl_img_cnt == num_images):
            batch_cnt += 1
            print '\n===> Processing batch #%5d with %5d images' % (batch_cnt, batch_img_cnt)

            last_new_id = process_image_list(prob_dir, prob_thresh,
                                             last_new_id, new_id_map,
                                             calc_orig_label_prob,
                                             output_fp1, output_fp2,
                                             img_list, label_list
                                             )
            batch_img_cnt = 0
            img_list = []
            label_list = []

            output_fp1.flush()
            output_fp2.flush()

        if (num_images > 0 and ttl_img_cnt == num_images):
            break

    if batch_img_cnt > 0:
        batch_cnt += 1
        print '\n===> Processing batch #%5d with %5d images' % (batch_cnt, batch_img_cnt)
        last_new_id = process_image_list(prob_dir, prob_thresh,
                                         last_new_id, new_id_map,
                                         calc_orig_label_prob,
                                         output_fp1, output_fp2,
                                         img_list, label_list
                                         )

    fp.close()

    output_fp1.close()
    output_fp2.close()

    new_id_map_fn = 'prob_thresh_%g-nonoverlap-new_id_map.npy' % prob_thresh
    new_id_map_fn = osp.join(save_dir, new_id_map_fn)
    np.save(new_id_map_fn, new_id_map)


if __name__ == '__main__':

    prob_thresh = 0.6
    first_new_id = 10572
    max_orig_label = 2

    # image path: osp.join(image_dir, <each line in image_list_file>)
    prob_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/corr_prob'
    image_list_file = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'

    # save_dir = None
    save_dir = osp.join(prob_dir, '..')

    num_images = -1
    mirror_input = False

    load_probs_and_refine_labels(prob_dir, prob_thresh,
                                 first_new_id, max_orig_label,
                                 image_list_file, num_images,
                                 save_dir)
