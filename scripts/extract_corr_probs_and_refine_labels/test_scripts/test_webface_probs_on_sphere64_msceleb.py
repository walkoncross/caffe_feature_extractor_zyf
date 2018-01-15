#!/usr/bin/env python
import _init_paths
from extract_probs_and_refine_labels import extract_probs_and_refine_labels
import json
import sys

if __name__ == '__main__':

    config_json = './extractor_config_sphere64_msceleb_nfs.json'
    fp = open(config_json, 'r')
    config_json = json.load(fp)
    fp.close()

    if len(sys.argv) > 1:
        config_json['gpu_id'] = int(sys.argv[1])

    prob_thresh = 0.55
    first_new_id = 78771
    max_orig_label = 10571

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'/disk2/data/FACE/webface/CASIA-maxpy-clean_mtcnn_simaligned_96x112'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/webface/webface-aligned-list-10572-ids-450833-objs-170503-213839.txt'

    save_dir = 'webface_probs_on_sphere64_msceleb'
    num_images = 512
    mirror_input = False

    extract_probs_and_refine_labels(config_json, prob_thresh,
                                    first_new_id, max_orig_label,
                                    image_list_file, image_dir,
                                    save_dir, num_images, mirror_input)
