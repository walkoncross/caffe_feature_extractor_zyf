#!/usr/bin/env python
import _init_paths
from extract_and_save_probs_stats import extract_and_save_probs_stats
import json
import sys

if __name__ == '__main__':

    config_json = '../extractor_config_sphere64_webface_nfs.json'
    fp = open(config_json, 'r')
    config_json = json.load(fp)
    fp.close()

    if len(sys.argv) > 1:
        config_json['gpu_id'] = int(sys.argv[1])
		
    is_train_set = False
    max_orig_label = 10571

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'/disk2/data/FACE/webface/CASIA-maxpy-clean_mtcnn_simaligned_96x112'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/webface/webface-aligned-list-10572-ids-450833-objs-170503-213839.txt'

    save_dir = '../../prob-results/webface_probs_on_sphere64_webface'
    num_images = -1 # <0, means all images
    mirror_input = False

    extract_and_save_probs_stats(config_json, max_orig_label,
                                 image_list_file, image_dir,
                                 save_dir, num_images,
                                 is_train_set, mirror_input)
