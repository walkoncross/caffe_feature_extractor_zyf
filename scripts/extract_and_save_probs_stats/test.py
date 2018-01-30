#!/usr/bin/env python
from extract_and_save_probs_stats import extract_and_save_probs_stats

if __name__ == '__main__':

    config_json = './extractor_config_sphere64_webface.json'

    is_train_set = False
    max_orig_label = 2

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'../../test_data/face_chips'
    image_list_file = r'../../test_data/face_chips_list_with_label.txt'

    save_dir = 'rlt_probs_stats_test'
    num_images = -1
    mirror_input = False

    extract_and_save_probs_stats(config_json, max_orig_label,
                                 image_list_file, image_dir,
                                 save_dir, num_images,
                                 is_train_set, mirror_input)
