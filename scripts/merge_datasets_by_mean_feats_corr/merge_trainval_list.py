# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 08:24:25 2018

@author: zhaoy
"""

import os
import os.path as osp
import numpy as np


def load_new_id_map(fn):
    new_id_map = np.loadtxt(fn, skiprows=1)

    return new_id_map


def merge_trainval_list(trainval_fns, new_id_map_fns, save_fn, root_dirs=None):
    assert(len(trainval_fns) == len(new_id_map_fns))

    if root_dirs:
        assert(len(trainval_fns) == len(root_dirs))

    print 'Will save merge result into file: ', save_fn
    save_dir = osp.dirname(save_fn)
    if save_dir and not osp.exists(save_dir):
        os.makedirs(save_dir)

    fp_out = open(save_fn, 'w')

    for i, list_fn in enumerate(trainval_fns):
        fp = open(list_fn, 'r')

        root_dir = ''
        if root_dirs and root_dirs[i]:
            root_dir = root_dirs[i]
            if not root_dir.endwith('/') or not root_dir.endwith('\\'):
                root_dir += '/'

        if i == 0:
            for line in fp:
                fp_out.write(root_dir + line)

        else:
            new_id_map = load_new_id_map(new_id_map_fns[i])
            for line in fp:
                spl = line.split(line)
                img_fn = root_dir + spl[0]
                orig_id = int(spl[1])

                new_id = new_id_map[orig_id, 1]
                if new_id >= 0:
                    fp_out.write('%s %d\n' % (img_fn, new_id))


if __name__ == '__main__':
    # train list files
    root_dir1 = '/disk2/zhaoyafei/face-recog-train/train-val-lists'
    train_list_fns = [
        'msceleb-1m/msceleb_fixed2_train_list_ratio-0.95_78771-ids_4833609-objs_171208-230236_refined_thr0.5333-nonoverlap.txt',
        'asian/face_asian_train_list_ratio-0.95_10245-ids_518838-objs_170818-225309.txt',
        'webface/webface_train_list_ratio-0.95_10572-ids_433705-objs_171226-051743.txt',
        'vggface2/vggface2_train_list_ratio-0.95_8631-ids_2988783-objs_171115-224724.txt'
        # 'vggface/train_list_ratio-0.95_2564-ids_1643339-objs_170727-211449.txt'
    ]
    for it in train_list_fns:
        it = osp.join(root_dir1, it)

    # val list files
    val_list_fns = [
        'msceleb_fixed2_val_list_ratio-0.05_70111-ids_215196-objs_171208-230236_refined_thr0.5333-nonoverlap.txt',
        'asian/face_asian_val_list_ratio-0.05_9964-ids_21897-objs_170818-225309.txt',
        'webface/webface_val_list_ratio-0.05_6770-ids_17026-objs_171226-051743.txt',
        'vggface2/vggface2_val_list_ratio-0.05_8631-ids_152995-objs_171115-224724.txt'
        # 'vggface/val_list_ratio-0.05_2564-ids_85226-objs_170727-211449.txt'
    ]
    for it in train_list_fns:
        it = osp.join(root_dir1, it)

    # new id map files
    root_dir2 = './rlt_merge_id_map_delete_overlap/'
    new_id_map_fns = [
        '',
        'asian_merge_with_msceleb_new_id_map.txt',
        'webface_merge_with_msceleb_asian_new_id_map.txt',
        'vggface2_merge_with_msceleb_asian_webface_new_id_map.txt'
        # 'vggface_merge_with_msceleb_asian_webface_vggface2_new_id_map.txt',
    ]

    for it in new_id_map_fns:
        it = osp.join(root_dir2, it)

    # image root dirs
    img_root_dirs = [
        "/disk2/data/FACE/celeb-1m-mtcnn-aligned/msceleb_align/Faces-Aligned/",
        '',
        '',
        '',
        # ''
    ]

    train_save_fn = 'train_list_merged_msceleb_asian_webface_vggface2_0.95.txt'
    merge_trainval_list(train_list_fns, new_id_map_fns,
                        train_save_fn, img_root_dirs)

    val_save_fn = 'val_list_merged_msceleb_asian_webface_vggface2_0.05.txt'
    merge_trainval_list(train_list_fns, new_id_map_fns,
                        val_save_fn, img_root_dirs)
