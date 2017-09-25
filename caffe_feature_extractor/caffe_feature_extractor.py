# by zhaoyafei0210@gmail.com

import os
import os.path as osp

import numpy as np
# import scipy.io as sio
import skimage

import json
import time
import caffe

from collections import OrderedDict

class CaffeFeatureException(Exception):
    pass


class CaffeFeatureExtractor(object):
    def __init__(self, config_json):
        self.net = None
        self.blobs = None
        self.input_shape = None
        self.batch_size = None

        self.config = {
            #'network_prototxt': '/path/to/prototxt',
            #'network_caffemodel': '/path/to/caffemodel',
            #'data_mean': '/path/to/the/mean/file',
            #'feature_layer': 'fc5',
            'batch_size': 1,
            'input_scale': 1.0,
            'raw_scale': 1.0,
            'channel_swap': (2, 1, 0),
            'mirror_trick': 0,  # 0,None - will not use mirror_trick, 1 - eltavg (i.e.
                                # eltsum()*0.5), 2 - eltmax
            'image_as_grey': False
        }

        if isinstance(config_json, str):
            if osp.isfile(config_json):
                fp = open(config_json, 'r')
                _config = json.load(fp)
                fp.close()
            else:
                _config = json.loads(config_json)
        else:
            _config = config_json

        _config['network_prototxt'] = str(_config['network_prototxt'])
        _config['network_caffemodel'] = str(_config['network_caffemodel'])
        _config['data_mean'] = str(_config['data_mean'])
        _config['channel_swap'] = tuple([int(i) for i in _config['channel_swap'].split(',') ])

        self.config.update(_config)

        if (self.config['data_mean'] is not None and
                type(self.config['data_mean']) is str):
            self.config['data_mean'] = np.load(self.config['data_mean'])

        print 'CaffeFeatureExtractor.config: \n', self.config

        caffe.set_mode_gpu()

        self.net = caffe.Classifier(self.config['network_prototxt'],
                                    self.config['network_caffemodel'],
                                    None,
                                    self.config['data_mean'],
                                    self.config['input_scale'],
                                    self.config['raw_scale'],
                                    self.config['channel_swap']
                                    )

        self.blobs = OrderedDict([(k, v.data)
                                  for k, v in self.net.blobs.items()])

        self.input_shape = self.blobs['data'].shape
        self.batch_size = self.input_shape[0]
        if self.config['mirror_trick'] > 0:
            if self.batch_size < 2:
                raise CaffeFeatureException('CaffeFeatureExtractor Exception:'
                                            ' If using mirror_trick, batch_size of input "data" layer must > 1')

            self.batch_size /= 2
            print 'halve the batch_size for mirror_trick eval: batch_size=', self.batch_size

        print 'original input data shape: ', self.input_shape
        print 'original batch_size: ', self.batch_size

    def load_image(self, image_path):
        img = caffe.io.load_image(image_path, color=not self.config['image_as_grey'])
        if self.config['image_as_grey'] and img.shape[2] != 1:
            img = skimage.color.rgb2gray(img)
            img = img[:, :, np.newaxis]

        return img

    def load_images(self, image_path):
        pass

    def extract_feature(self, image, layer_name=None):
        if not layer_name:
            layer_name = self.config['feature_layer']

        if not layer_name:
            raise CaffeFeatureException('CaffeFeatureExtractor Exception:'
                                        ' Invalid layer_name')

        shp = self.blobs[layer_name].shape
        print 'feature layer shape: ', shp

        img_batch = []
        cnt_load_img = 0
        cnt_predict = 0

        time_load_img = 0.0
        time_predict = 0.0

        if isinstance(image, str):
            t1 = time.clock()
            img = self.load_image(image)
            cnt_load_img += 1
            t2 = time.clock()
            time_predict += (t2 - t1)
        else:
            img = image
        print 'image shape: ', img.shape

        img_batch.append(img)

        if self.config['mirror_trick']:
            mirror_img = np.fliplr(img)
            img_batch.append(mirror_img)
            print 'add mirrored images into predict batch'
            print 'after add: len(img_batch)=%d' % (len(img_batch))

        n_imgs = 1
        t1 = time.clock()

        self.net.predict(img_batch, oversample=False)

        t2 = time.clock()
        time_predict += (t2 - t1)
        cnt_predict += n_imgs

        # must call blobs_data(v) again, because it invokes (mutable_)cpu_data() which
        # syncs the memory between GPU and CPU
        blobs = OrderedDict([(k, v.data)
                             for k, v in self.net.blobs.items()])

        if self.config['mirror_trick']:
            ftrs = blobs[layer_name][0:n_imgs * 2, ...]
            if self.config['mirror_trick'] == 2:
                eltop_ftrs = np.maximum(ftrs[:n_imgs], ftrs[n_imgs:])
            else:
                eltop_ftrs = (ftrs[:n_imgs] + ftrs[n_imgs:]) * 0.5

            feature = eltop_ftrs[0]

        else:
            feature = blobs[layer_name][0, ...]

        if cnt_load_img:
            print ('load %d images cost %f seconds, average time: %f seconds'
                   % (cnt_load_img, time_load_img, time_load_img / cnt_load_img))

        print ('predict %d images cost %f seconds, average time: %f seconds'
               % (cnt_predict, time_predict, time_predict / cnt_predict))

        feature = np.asarray(feature, dtype='float32')

        return feature

    def extract_features_batch(self, images, layer_name=None):
        if not layer_name:
            layer_name = self.config['feature_layer']

        if not layer_name:
            raise CaffeFeatureException('CaffeFeatureExtractor Exception:'
                                        ' Invalid layer_name')

        n_imgs = len(images)

        if (n_imgs > self.batch_size
                or (self.config['mirror_trick'] and n_imgs / 2 > self.batch_size)):
            raise CaffeFeatureException('CaffeFeatureExtractor Exception:'
                                        ' Number of input images > batch_size set in prototxt')

        shp = self.blobs[layer_name].shape
        print 'feature layer shape: ', shp

        features_shape = (len(images),) + shp[1:]
        features = np.empty(features_shape, dtype='float32', order='C')
        print 'output features shape: ', features_shape

        img_batch = images

        cnt_predict = 0
        time_predict = 0.0

        if self.config['mirror_trick'] > 0:
            for i in range(n_imgs):
                mirror_img = np.fliplr(img_batch[i])
                img_batch.append(mirror_img)
            print 'add mirrored images into predict batch'
            print 'after add: len(img_batch)=%d' % (len(img_batch))

        t1 = time.clock()

        self.net.predict(img_batch, oversample=False)

        t2 = time.clock()
        time_predict += (t2 - t1)
        cnt_predict += n_imgs

        # must call blobs_data(v) again, because it invokes (mutable_)cpu_data() which
        # syncs the memory between GPU and CPU
        blobs = OrderedDict([(k, v.data)
                             for k, v in self.net.blobs.items()])

        if self.config['mirror_trick']:
            ftrs = blobs[layer_name][0:n_imgs * 2, ...]
            if self.config['mirror_trick'] == 2:
                eltop_ftrs = np.maximum(ftrs[:n_imgs], ftrs[n_imgs:])
            else:
                eltop_ftrs = (ftrs[:n_imgs] + ftrs[n_imgs:]) * 0.5

            features = eltop_ftrs

        else:
            ftrs = blobs[layer_name][0:n_imgs, ...]
            features = ftrs

        print('Predict %d images, cost %f seconds, average time: %f seconds' %
              (cnt_predict, time_predict, time_predict / cnt_predict))

        features = np.asarray(features, dtype='float32')

        return features

    def extract_features_for_image_list(self, image_list, layer_name=None):
        if not layer_name:
            layer_name = self.config['feature_layer']

        if not layer_name:
            raise CaffeFeatureException('CaffeFeatureExtractor Exception:'
                                        ' Invalid layer_name')

        shp = self.blobs[layer_name].shape
        print 'feature layer shape: ', shp

        features_shape = (len(image_list),) + shp[1:]
        features = np.empty(features_shape, dtype='float32', order='C')
        print 'output features shape: ', features_shape
        img_batch = []

        cnt_load_img = 0
        time_load_img = 0.0
#        cnt_predict = 0
#        time_predict = 0.0

        for cnt, path in zip(range(features_shape[0]), image_list):
            t1 = time.clock()
            img = self.load_image(path)
            if cnt == 0:
                print 'image shape: ', img.shape

            img_batch.append(img)
            t2 = time.clock()

            cnt_load_img += 1
            time_load_img += (t2 - t1)

            # print 'image shape: ', img.shape
            # print path, type(img), img.mean()
            if (len(img_batch) == self.batch_size) or cnt == features_shape[0] - 1:
                # n_imgs = len(img_batch)
                # if self.config['mirror_trick'] > 0:
                #     for i in range(n_imgs):
                #         mirror_img = np.fliplr(img_batch[i])
                #         img_batch.append(mirror_img)
                #     print 'add mirrored images into predict batch'
                #     print 'after add: len(img_batch)=%d' % (len(img_batch))

                # t1 = time.clock()

                # self.net.predict(img_batch, oversample=False)

                # t2 = time.clock()
                # time_predict += (t2 - t1)
                # cnt_predict += n_imgs

                # # must call blobs_data(v) again, because it invokes (mutable_)cpu_data() which
                # # syncs the memory between GPU and CPU
                # blobs = OrderedDict([(k, v.data)
                #                      for k, v in self.net.blobs.items()])

                # print ('predict %d images cost %f seconds, average time: %f seconds'
                #        % (cnt_predict, time_predict, time_predict / cnt_predict))
                # print '%d images processed' % (cnt + 1,)

                # if self.config['mirror_trick']:
                #     ftrs = blobs[layer_name][0:n_imgs * 2, ...]
                #     if self.config['mirror_trick'] == 2:
                #         eltop_ftrs = np.maximum(ftrs[:n_imgs], ftrs[n_imgs:])
                #     else:
                #         eltop_ftrs = (ftrs[:n_imgs] + ftrs[n_imgs:]) * 0.5

                #     features[cnt - n_imgs + 1:cnt + 1, ...] = eltop_ftrs

                # else:
                #     ftrs = blobs[layer_name][0:n_imgs, ...]
                #     features[cnt - n_imgs + 1:cnt + 1, ...] = ftrs.copy()
                n_imgs = len(img_batch)
                ftrs = self.extract_features_batch(img_batch, layer_name)
                features[cnt - n_imgs + 1:cnt + 1, ...] = ftrs.copy()

                img_batch = []

        print('Load %d images, cost %f seconds, average time: %f seconds' %
              (cnt_load_img, time_load_img, time_load_img / cnt_load_img))
#        print('Predict %d images, cost %f seconds, average time: %f seconds' %
#              (cnt_predict, time_predict, time_predict / cnt_predict))

        features = np.asarray(features, dtype='float32')

        return features


if __name__ == '__main__':
    def load_image_list(img_dir, list_file_name):
        #list_file_path = os.path.join(img_dir, list_file_name)
        f = open(list_file_name, 'r')
        image_fullpath_list = []

        for line in f:
            if line.startswith('#'):
                continue

            items = line.split()
            image_fullpath_list.append(os.path.join(img_dir, items[0].strip()))

        f.close()

        return image_fullpath_list

    ## init a feat_extractor
    print '\n===> init a feat_extractor'
    config_json = './extractor_config.json'
    feat_extractor = CaffeFeatureExtractor(config_json)

    ## test extract_features_for_image_list()
    image_dir = r'C:\zyf\github\mtcnn-caffe-good\face_aligner\face_chips'
    image_list_file = r'C:\zyf\github\lfw-evaluation-zyf\extract_face_features\face_chips\face_chips_list_2.txt'
    save_name = 'img_list_features_eltavg.npy'

    img_list = load_image_list(image_dir, image_list_file)

    print '\n===> test extract_features_for_image_list()'
    ftrs = feat_extractor.extract_features_for_image_list(img_list)
    np.save(save_name, ftrs)

    for i in range(len(img_list)):
        save_name = osp.splitext(osp.basename(img_list[i]))[0] + '_feat_eltavg.npy'
        np.save(save_name, ftrs[i])

    ## test extract_feature()
    print '\n===> test extract_feature()'
    save_name_2 = 'single_feature_eltavg.npy'
    ftr = feat_extractor.extract_feature(img_list[0])
    np.save(save_name_2, ftr)

    ft_diff = ftr - ftrs[0]
    print 'ft_diff: ', ft_diff.sum()