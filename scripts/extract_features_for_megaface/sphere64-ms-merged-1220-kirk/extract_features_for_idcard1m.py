#! /usr/bin/env python
import _init_paths
import sys
from extract_features import extract_features


if __name__ == '__main__':
    config_json = './extractor_config_sphere64_pod.json'
    save_dir = '/disk2/data/FACE/features/sphereface-64-1220/IDCARD1M_SPF512Features'
    gpu_id = None

    if len(sys.argv) > 1:
        gpu_id = int(sys.argv[1])

    # image path: osp.join(image_dir, <each line in image_list_file>)
    image_dir = r'/disk2/data/FACE/face-idcard-1M/face-idcard-1M-mtcnn-aligned-96x112/aligned_imgs'
    image_list_file = r'/disk2/data/FACE/face-idcard-1M/face-idcard-1M-image-list.txt'
    extract_features(config_json, save_dir, image_list_file, image_dir, gpu_id=gpu_id)
