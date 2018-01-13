from extract_probs_and_refine_labels import extract_probs_and_refine_labels

if __name__ == '__main__':

    config_json = './extractor_config_sphere64_webface.json'

    prob_thresh = 0.5
    first_new_id = 0

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'C:\zyf\github\mtcnn-caffe-good-new\face_aligner\face_chips'
    image_list_file = r'.\face_chips_list_with_label.txt'

    save_dir = 'rlt_probs_and_refined_labels'
    num_images = -1

    extract_probs_and_refine_labels(config_json, prob_thresh, first_new_id,
                                    image_list_file, image_dir,
                                    save_dir, num_images)
