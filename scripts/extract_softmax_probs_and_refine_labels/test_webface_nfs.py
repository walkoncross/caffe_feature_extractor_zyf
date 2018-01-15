#!/usr/bin/env python
from extract_probs_and_refine_labels import extract_probs_and_refine_labels

if __name__ == '__main__':

    config_json = './extractor_config_sphere64_webface_nfs.json'

    prob_thresh = 0.7
    first_new_id = 0
    max_orig_label = 10572

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'/disk2/data/FACE/webface/CASIA-maxpy-clean_mtcnn_simaligned_96x112'
    image_list_file = r'/disk2/zhaoyafei/face-recog-train/train-val-lists/webface/webface-aligned-list-10572-ids-450833-objs-170503-213839.txt'

    save_dir = 'webface_probs_and_refined_labels'
    num_images = 512 # <0, means all images
    mirror_input = False

    extract_probs_and_refine_labels(config_json, prob_thresh,
                                    first_new_id, max_orig_label,
                                    image_list_file, image_dir,
                                    save_dir, num_images, mirror_input)
