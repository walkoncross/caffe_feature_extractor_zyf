#!/usr/bin/env python
from extract_and_save_probs_stats import extract_and_save_probs_stats

if __name__ == '__main__':

    config_json = './extractor_config_sphere64_webface_nfs.json'

    is_train_set = True
    max_orig_label = 10571

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'/disk2/data/FACE/webface/CASIA-maxpy-clean_mtcnn_simaligned_96x112'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/webface/webface-aligned-list-10572-ids-450833-objs-170503-213839.txt'

    save_dir = './webface_probs_and_refined_labels'
    num_images = 512 # <0, means all images
    mirror_input = False

    extract_and_save_probs_stats(config_json, max_orig_label,
                                 image_list_file, image_dir,
                                 save_dir, num_images,
                                 is_train_set, mirror_input)
