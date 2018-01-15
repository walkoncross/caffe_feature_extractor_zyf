#!/usr/bin/env python
import _init_paths
from extract_probs_and_refine_labels import extract_probs_and_refine_labels
import json
import sys

if __name__ == '__main__':

    config_json = '../extractor_config_sphere64_webface_nfs.json'
    fp = open(config_json, 'r')
    config_json = json.load(fp)
    fp.close()

    if len(sys.argv) > 1:
        config_json['gpu_id'] = int(sys.argv[1])
		
    prob_thresh = 0.55
    first_new_id = 0
    max_orig_label = 10572

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'/disk2/data/FACE/face-asian/face_asian_align/'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/asian/face_asian_train_list_noval_10245-ids_540735-objs_170818-225846-norootdir.txt'

    save_dir = 'asian_probs_on_sphere64_webface'
    num_images = -1
    mirror_input = False

    extract_probs_and_refine_labels(config_json, prob_thresh,
                                    first_new_id, max_orig_label,
                                    image_list_file, image_dir,
                                    save_dir, num_images, mirror_input)
