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
    first_new_id = 10572
    max_orig_label = 8630

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'/disk2/data/FACE/vggface2/vggface2_train_aligned/aligned_imgs/'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/vggface2/vggface2_aligned_list_8631-ids_3141890-objs_171103-200428.txt'

    save_dir = '../../prob-results/vggface2_probs_on_sphere64_webface'
    num_images = -1 # <0, means all images
    mirror_input = False

    extract_probs_and_refine_labels(config_json, prob_thresh,
                                    first_new_id, max_orig_label,
                                    image_list_file, image_dir,
                                    save_dir, num_images, mirror_input)
