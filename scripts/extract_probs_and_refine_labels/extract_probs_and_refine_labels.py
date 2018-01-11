import os
import os.path as osp
import numpy as np
import json

import _init_paths
from caffe_feature_extractor import CaffeFeatureExtractor


PROB_THRESH = 0.7

FIRST_NEW_ID = 78771


def process_image_list(feat_extractor, img_list, out_fp1, out_fp2, label_list=None, image_dir=None):
    ftrs = feat_extractor.extract_features_for_image_list(img_list, image_dir)
#    np.save(osp.join(save_dir, save_name), ftrs)

    # root_len = len(image_dir)

    for i in range(len(img_list)):
        spl = osp.split(img_list[i])
        base_name = spl[1]
#        sub_dir = osp.split(spl[0])[1]
        sub_dir = spl[0]
        save_sub_dir = save_dir

        if sub_dir:
            save_sub_dir = osp.join(save_dir, sub_dir)
            if not osp.exists(save_sub_dir):
                os.makedirs(save_sub_dir)

        probs = np.ravel(ftrs[i])
        print 'probs.shape:', probs.shape
        save_name = osp.splitext(base_name)[0] + '.npy'
        np.save(osp.join(save_sub_dir, save_name), probs)

        print '---> image: ' + img_list[i]
        if label_list:
            print 'original label: ', label_list[i]

        max_label = np.argmax(probs)
        print 'max_label in probs: ', probs[i]

        if probs[max_label] < PROB_THRESH:
            if out_fp1:
                if label_list:
                    new_label = label_list[i] + FIRST_NEW_ID
                    write_string = "%s\t%d\t%d\t%g\t%d\n" % (
                        img_list[i], new_label, max_label, probs[max_label], label_list[i])
                else:
                    new_label = -1
                    write_string = "%s\t%d\t%d\t%g\n" % (
                        img_list[i], new_label, max_label, probs[max_label])
                out_fp1.write(write_string)
        else:
            new_label = max_label

            if out_fp2:
                if label_list:
                    write_string = "%s\t%d\t%d\t%g\t%d\n" % (
                        img_list[i], new_label, max_label, probs[max_label], label_list[i])
                else:
                    write_string = "%s\t%d\t%d\t%g\n" % (
                        img_list[i], new_label, max_label, probs[max_label])
                out_fp2.write(write_string)


if __name__ == '__main__':
    config_json = './extractor_config_sphere64_webface.json'
    save_dir = 'refined_probs_and_labels'

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'C:\zyf\github\mtcnn-caffe-good-new\face_aligner\face_chips'
    image_list_file = r'.\face_chips_list_with_label.txt'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # test extract_features_for_image_list()
    output_fn1 = 'img_list_nonoverlap.txt'
    output_fp1 = open(osp.join(save_dir, output_fn1), 'w')
    output_fn2 = 'img_list_overlap.txt'
    output_fp2 = open(osp.join(save_dir, output_fn2), 'w')

    fp = open(image_list_file, 'r')

    # init a feat_extractor
    print '\n===> init a feat_extractor'

    feat_extractor = CaffeFeatureExtractor(config_json)
    batch_size = feat_extractor.get_batch_size()
    # overwrite 'feature_layer' in config by the last layer's name
    prob_layer = feat_extractor.get_final_layer_name()
    print '===> prob layer name: ', prob_layer
    feat_extractor.set_feature_layer(prob_layer)

    print 'feat_extractor can process %d images in a batch' % batch_size

    img_list = []
    label_list = []
    cnt = 0
    batch_cnt = 0

    for line in fp:
        if line.startswith('#'):
            continue

        spl = line.split()
        img_list.append(spl[0].strip())

        if (len(spl) > 1):
            label_list.append(int(spl[1]))

        cnt += 1

        if cnt == batch_size:
            batch_cnt += 1
            print '\n===> Processing batch #%d with %d images' % (batch_cnt, cnt)

            process_image_list(feat_extractor, img_list,
                               output_fp1, output_fp2, label_list, image_dir)
            cnt = 0
            img_list = []
            label_list = []

    if cnt > 0:
        batch_cnt += 1
        print '\n===> Processing batch #%d with %d images' % (batch_cnt, cnt)
        process_image_list(feat_extractor, img_list,
                           output_fp1, output_fp2, label_list, image_dir)

    fp.close()

    output_fp1.close()
    output_fp2.close()