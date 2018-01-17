#!/usr/bin/env python
import sys
import os
import os.path as osp
import numpy as np
import json

import _init_paths
from caffe_feature_extractor import CaffeFeatureExtractor

from numpy.linalg import norm


def process_image_batch(feat_extractor,
                        cnt_per_id_vec,
                        feats_stats_dict,
                        img_list, label_list=None,
                        image_dir=None,
                        mirror_input=False):

    ftrs = feat_extractor.extract_features_for_image_list(
        img_list, image_dir, mirror_input=mirror_input)
#    np.save(osp.join(save_dir, save_name), ftrs)
    feat_layer_names = feat_extractor.get_feature_layers()

    feat_layer = feat_layer_names[0]
    corr_prob_layer = feat_layer_names[1]

    output_corr_dir = 'corr_prob'

    num_ids = len(cnt_per_id_vec)

    for layer in feat_layer_names:
        shp = (2, num_ids,) + ftrs[layer].shape[1:]
        if (layer is corr_prob_layer and
                output_corr_dir not in feats_stats_dict.keys()):
            feats_stats_dict[output_corr_dir] = np.zeros(
                shp, dtype=ftrs[layer].dtype)
        elif layer not in feats_stats_dict.keys():
            feats_stats_dict[layer] = np.zeros(
                shp, dtype=ftrs[layer].dtype)

    for i in range(len(img_list)):
        cnt_per_id_vec[label_list[i]] += 1

        for layer in feat_layer_names:
            if layer is corr_prob_layer:
                continue

            feats_stats_dict[layer][0, label_list[i]] += ftrs[layer][i]
            feats_stats_dict[layer][1, label_list[i]
                                    ] += np.square(ftrs[layer][i])

        # calculate correlation probs
        feat = np.ravel(ftrs[feat_layer][i])
        feat_norm = norm(feat)
        probs = np.ravel(ftrs[corr_prob_layer][i]) / feat_norm

        feats_stats_dict[output_corr_dir][0, label_list[i]] += probs
        feats_stats_dict[output_corr_dir][1, label_list[i]] += np.square(probs)

        return feats_stats_dict


def save_max_label_info_stats(cnt_per_id_fn,
                              feats_stats_dict,
                              save_dir, is_train_dataset=False):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    num_ids = cnt_per_id_fn.shape[0]

    for layer in feats_stats_dict.keys():
        if 'prob' not in layer:
            continue

        probs_sum_vec = feats_stats_dict[layer][0]
        probs_sqsum_vec = feats_stats_dict[layer][1]

        output_fn = layer + '-stats-max-label-info.txt'
        output_fp = open(osp.join(save_dir, output_fn), 'w')

        write_string = 'orig_label max_label  probs_avg[max_label]  probs_std[max_label]'
        if is_train_dataset:
            write_string += '  probs_avg[orig_label]  probs_std[orig_label]'
        write_string += '\n'
        output_fp.write(write_string)

        inter_ids_avg_min = np.ones(2) * 1.0e10
        inter_ids_avg_avg = np.ones(2) * 0
        inter_ids_avg_max = np.ones(2) * -1.0e10

        inter_ids_std_min = np.ones(2) * 1.0e10
        inter_ids_std_avg = np.ones(2) * 0
        inter_ids_std_max = np.ones(2) * -1.0e10

        for i in range(num_ids):
            max_label = np.argmax(probs_sum_vec[i])

            inter_ids_avg_min[0] = min(
                inter_ids_avg_min[0], probs_sum_vec[i][max_label])
            inter_ids_avg_max[0] = max(
                inter_ids_avg_max[0], probs_sum_vec[i][max_label])
            inter_ids_avg_avg[0] += probs_sum_vec[i][max_label]

            inter_ids_std_min[0] = min(
                inter_ids_std_min[0], probs_sqsum_vec[i][max_label])
            inter_ids_std_max[0] = max(
                inter_ids_std_max[0], probs_sqsum_vec[i][max_label])
            inter_ids_std_avg[0] += probs_sqsum_vec[i][max_label]

            write_string = '%d    %d    %.4f    %.4f' % (
                i, max_label, probs_sum_vec[i][max_label], probs_sqsum_vec[i][max_label], )

            if is_train_dataset:
                write_string += '    %.4f    %.4f' % (
                    probs_sum_vec[i][i], probs_sqsum_vec[i][i])

                inter_ids_avg_min[1] = min(
                    inter_ids_avg_min[1], probs_sum_vec[i][i])
                inter_ids_avg_max[1] = max(
                    inter_ids_avg_max[1], probs_sum_vec[i][i])
                inter_ids_avg_avg[1] += probs_sum_vec[i][i]

                inter_ids_std_min[1] = min(
                    inter_ids_std_min[1], probs_sqsum_vec[i][i])
                inter_ids_std_max[1] = max(
                    inter_ids_std_max[1], probs_sqsum_vec[i][i])
                inter_ids_std_avg[1] += probs_sqsum_vec[i][i]

            write_string += '\n'
            output_fp.write(write_string)

        write_string = 'min    ---    %.4f    %.4f' % (
            inter_ids_avg_min[0], inter_ids_std_min[0])
        if is_train_dataset:
            write_string += '    %.4f    %.4f' % (
                inter_ids_avg_min[1], inter_ids_std_min[1])
        write_string += '\n'
        output_fp.write(write_string)

        write_string = 'max    ---    %.4f    %.4f' % (
            inter_ids_avg_max[0], inter_ids_std_max[0])
        if is_train_dataset:
            write_string += '    %.4f    %.4f' % (
                inter_ids_avg_max[1], inter_ids_std_max[1])
        write_string += '\n'
        output_fp.write(write_string)

        write_string = 'avg    ---    %.4f    %.4f' % (
            inter_ids_avg_avg[0] / num_ids, inter_ids_std_avg[0] / num_ids)
        if is_train_dataset:
            write_string += '    %.4f    %.4f' % (
                inter_ids_avg_avg[1] / num_ids, inter_ids_std_avg[1] / num_ids)
        write_string += '\n'
        output_fp.write(write_string)

        output_fp.close()


def save_feats_stats_dict(feats_stats_dict, save_dir):
    for layer in feats_stats_dict.keys():
        save_name = layer + '_feat_avg_for_ids.npy'
        np.save(osp.join(save_dir, save_name), feats_stats_dict[layer][0])

        save_name = layer + '_feat_std_for_ids.npy'
        np.save(osp.join(save_dir, save_name), feats_stats_dict[layer][1])


def extract_and_save_probs_stats(config_json, max_orig_label,
                                 image_list_file, image_dir,
                                 save_dir=None, num_images=-1,
                                 is_train_set=False,
                                 mirror_input=False):

    if not save_dir:
        save_dir = 'rlt_probs_stats'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    num_ids = max_orig_label + 1

    fp = open(image_list_file, 'r')

    # init a feat_extractor
    print('\n===> init a feat_extractor')

    feat_extractor = CaffeFeatureExtractor(config_json)
    batch_size = feat_extractor.get_batch_size()
    # overwrite 'feature_layer' in config by the last layer's name
    # prob_layer = feat_extractor.get_final_layer_name()
    # print '===> prob layer name: ', prob_layer
    # feat_extractor.set_feature_layers(prob_layer)

    print('feat_extractor can process %5d images in a batch' % batch_size)

    img_list = []
    label_list = []

    ttl_img_cnt = 0
    batch_img_cnt = 0
    batch_cnt = 0

    cnt_per_id_vec = np.zeros(num_ids, dtype=np.int32)
    feats_stats_dict = {}

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
            print('\n===> Processing batch #%5d with %5d images' %
                  (batch_cnt, batch_img_cnt))

            feats_stats_dict = process_image_batch(feat_extractor,
                                                   cnt_per_id_vec,
                                                   feats_stats_dict,
                                                   img_list, label_list,
                                                   image_dir, mirror_input)
            batch_img_cnt = 0
            img_list = []
            label_list = []

        if (num_images > 0 and ttl_img_cnt == num_images):
            break

    if batch_img_cnt > 0:
        batch_cnt += 1
        print('\n===> Processing batch #%5d with %5d images' %
              (batch_cnt, batch_img_cnt))
        feats_stats_dict = process_image_batch(feat_extractor,
                                               cnt_per_id_vec,
                                               feats_stats_dict,
                                               img_list, label_list,
                                               image_dir, mirror_input)

    fp.close()

    cnt_per_id_fn = osp.join(save_dir, 'stats-cnt_per_id.npy')
    np.save(cnt_per_id_fn, cnt_per_id_vec)

    for layer in feats_stats_dict.keys():
        for i in range(num_ids):
            if cnt_per_id_vec[i] > 0:
                # sum -> avg
                feats_stats_dict[layer][0, i] /= cnt_per_id_vec[i]
                # sqsum -> std
                feats_stats_dict[layer][1, i] /= cnt_per_id_vec[i]
                feats_stats_dict[layer][1,
                                        i] -= np.square(feats_stats_dict[layer][0, i])

    print('\n===> Saving features stats (avg, std) for each id under ' + save_dir)
    save_feats_stats_dict(feats_stats_dict, save_dir)

    print('\n===> Saving save_max_label_info_stats under ' + save_dir)
    save_max_label_info_stats(
        cnt_per_id_vec, feats_stats_dict, save_dir, is_train_set)


if __name__ == '__main__':

    config_json = './extractor_config_sphere64_webface.json'

    is_train_set = False
    max_orig_label = 2

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'C:\zyf\github\mtcnn-caffe-good-new\face_aligner\face_chips'
    image_list_file = r'.\face_chips_list_with_label.txt'

    save_dir = 'rlt_probs_stats'
    num_images = -1
    mirror_input = False

    extract_and_save_probs_stats(config_json, max_orig_label,
                                 image_list_file, image_dir,
                                 save_dir, num_images,
                                 is_train_set, mirror_input)
