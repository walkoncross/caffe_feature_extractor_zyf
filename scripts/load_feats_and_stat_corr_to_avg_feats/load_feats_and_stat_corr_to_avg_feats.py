#!/usr/bin/env python
import os
import os.path as osp
import numpy as np

from numpy.linalg import norm


def load_feature(prob_dir, img_fn):
    feat_fn = osp.splitext(img_fn)[0] + '.npy'
    feat_fn = osp.join(prob_dir, feat_fn)

    feat = np.load(feat_fn)
    norm_1 = norm(feat)
    if norm_1 > 0:
        feat /= norm_1

    return feat


def load_feature_list(prob_dir, img_list):
    probs_list = []
    for fn in img_list:
        feat = load_feature(prob_dir, fn)
        probs_list.append(feat)

    return probs_list


def load_avg_features(feat_fn):
    feat_set = np.load(feat_fn)

    for i in range(feat_set.shape[0]):
        norm_1 = norm(feat_set[i])
        if norm_1 > 0:
            feat_set[i] /= norm_1

    return feat_set


def calc_corr_to_avg_feats(feats, avg_feat_set):
    corr_mat12 = np.dot(feats, avg_feat_set.T)

    return corr_mat12


def load_feats_and_stat_corr_to_avg_feats(img_list_fn, feats_dir,
                                          avg_feat_fn, save_dir=None,
                                          num_images=-1, force_recalc=False):
    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    avg_feat_set = load_avg_features(avg_feat_fn)

    num_ids = avg_feat_set.shape[0]

    cnt_per_id_vec = np.zeros(num_ids, dtype=np.int32)

    probs_sum_vec = np.zeros((num_ids, num_ids), dtype=np.float32)
    probs_sqsum_vec = np.zeros((num_ids, num_ids), dtype=np.float32)

    fp = open(img_list_fn, 'r')

    batch_size = 512

    ttl_img_cnt = 0

    for line in fp:
        if line.startswith('#'):
            continue

        spl = line.split()
        img_fn = spl[0].strip()
        orig_label = int(spl[1])

        spl2 = osp.split(img_fn)
        sub_dir = osp.join(save_dir, spl2[0])
        if not osp.exists(sub_dir):
            os.makedirs(sub_dir)
        npy_fn = osp.join(sub_dir, osp.splitext(spl2[1])[0] + '.npy')

        ttl_img_cnt += 1

        loaded = False
        if osp.exists(npy_fn) and not force_recalc:
            # print('\n===> corr_to_avg_feats already exists, will load it')
            try:
                probs = np.load(npy_fn)
                loaded = True
            except:
                # if the .npy file is broken
                print('===> Failed to load: ' + npy_fn)
                print('will recalc it')
        if not loaded:
            feat = load_feature(feats_dir, img_fn)

            probs = calc_corr_to_avg_feats(feat, avg_feat_set)
            np.save(npy_fn, probs)

        probs_sum_vec[orig_label] += probs.T
        probs_sqsum_vec[orig_label] += np.square(probs.T)

        cnt_per_id_vec[orig_label] += 1

        if (np.mod(ttl_img_cnt, batch_size) == 0 or
                (num_images > 0 and ttl_img_cnt == num_images)):
            print('\n===> Processed %5d images' % (ttl_img_cnt))

        if (num_images > 0 and ttl_img_cnt == num_images):
            break

    fp.close()

    cnt_per_id_fn = osp.join(save_dir, 'stats-cnt_per_id.npy')
    np.save(cnt_per_id_fn, cnt_per_id_vec)

    output_fn = 'stats-max-label-info.txt'
    output_fp = open(osp.join(save_dir, output_fn), 'w')

    write_string = 'orig_label max_label  probs_avg[max_label]  probs_std[max_label]'
    write_string += '  probs_avg[orig_label]  probs_std[orig_label]'
    write_string += '  num_objs'
    write_string += '\n'
    output_fp.write(write_string)

    inter_ids_avg_min = np.ones(2) * 1.0e10
    inter_ids_avg_avg = np.ones(2) * 0
    inter_ids_avg_max = np.ones(2) * -1.0e10

    inter_ids_std_min = np.ones(2) * 1.0e10
    inter_ids_std_avg = np.ones(2) * 0
    inter_ids_std_max = np.ones(2) * -1.0e10

    for i in range(num_ids):
        if cnt_per_id_vec[i]:
            probs_sum_vec[i] /= cnt_per_id_vec[i]
            probs_sqsum_vec[i] /= cnt_per_id_vec[i]
            probs_sqsum_vec[i] -= np.square(probs_sum_vec[i])

            max_label = np.argmax(probs_sum_vec[i])
        else:
            max_label = -1

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
            i, max_label, probs_sum_vec[i][max_label], probs_sqsum_vec[i][max_label])

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

        write_string += '    %d' % (cnt_per_id_vec[i])
        write_string += '\n'
        output_fp.write(write_string)

    write_string = 'min    ---    %.4f    %.4f' % (
        inter_ids_avg_min[0], inter_ids_std_min[0])
    write_string += '    %.4f    %.4f' % (
        inter_ids_avg_min[1], inter_ids_std_min[1])
    write_string += '\n'
    output_fp.write(write_string)

    write_string = 'max    ---    %.4f    %.4f' % (
        inter_ids_avg_max[0], inter_ids_std_max[0])
    write_string += '    %.4f    %.4f' % (
        inter_ids_avg_max[1], inter_ids_std_max[1])
    write_string += '\n'
    output_fp.write(write_string)

    write_string = 'avg    ---    %.4f    %.4f' % (
        inter_ids_avg_avg[0] / num_ids, inter_ids_std_avg[0] / num_ids)
    write_string += '    %.4f    %.4f' % (
        inter_ids_avg_avg[1] / num_ids, inter_ids_std_avg[1] / num_ids)
    write_string += '\n'
    output_fp.write(write_string)

    output_fp.close()

    probs_avg_fn = osp.join(save_dir, 'stats-probs_avg_vec.npy')
    probs_std_fn = osp.join(save_dir, 'stats-probs_std_vec.npy')

    np.save(probs_avg_fn, probs_sum_vec)
    np.save(probs_std_fn, probs_sqsum_vec)

    return cnt_per_id_vec, probs_sum_vec, probs_sqsum_vec


if __name__ == '__main__':
    img_list_fn = '../extract_corr_probs_and_refine_labels/face_chips_list_with_label.txt'
    feat_dir = '../extract_corr_probs_and_refine_labels/rlt_probs_and_refined_labels/fc5/'
    avg_feat_fn = '../extract_and_save_probs_stats/rlt_probs_stats/fc5_feat_avg_for_ids.npy'
    num_images = -1
    save_dir = './rlt_stats_feats_corr_to_avg_feats'
    force_recalc = False

    load_feats_and_stat_corr_to_avg_feats(img_list_fn, feat_dir,
                                          avg_feat_fn, save_dir,
                                          num_images, force_recalc)
