#!/bin/usr/env python

# by zhaoyafei0210@gmail.com

import os
import os.path as osp

import numpy as np
from numpy.linalg import norm
# import scipy.io as sio
import skimage

import json
import time
import caffe

# from caffe import Classifier
from classifier import Classifier

# from collections import OrderedDict


def load_binaryproto(bp_file):
    blob_proto = caffe.proto.caffe_pb2.BlobProto()
    data = open(bp_file, 'rb').read()
    blob_proto.ParseFromString(data)
    arr = caffe.io.blobproto_to_array(blob_proto)
#    print type(arr)
    return arr


class Error(Exception):
    """Base class for exceptions in this module."""
    pass


class InitError(Error):
    """ Class for Init exceptions in this module."""
    pass


class FeatureLayerError(Error):
    """Exception for Invalid feature layer name."""
    pass


class ExtractionError(Error):
    """Exception from extract_xxx()."""
    pass


class CaffeFeatureExtractor(object):
    def __init__(self, config_json):
        self.net = None
#        self.net_blobs = None
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
            # default is BGR, be careful of your input image's channel
            'channel_swap': (2, 1, 0),
            # 0,None - will not use mirror_trick, 1 - eltavg (i.e.
            # eltsum()*0.5), 2 - eltmax
            'mirror_trick': 0,
            'image_as_grey': False,
            'normalize_output': False,
            'cpu_only': 0,
            'gpu_id': 0
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

        # must convert to str, because json.load() outputs unicode which is not support
        # in caffe's cpp function
        _config['network_prototxt'] = str(_config['network_prototxt'])
        _config['network_caffemodel'] = str(_config['network_caffemodel'])
        _config['data_mean'] = str(_config['data_mean'])
        _config['channel_swap'] = tuple(
            [int(i) for i in _config['channel_swap'].split(',')])

        self.config.update(_config)

        mean_arr = None
        if (self.config['data_mean']):
            try:
                if self.config['data_mean'].endswith('.npy'):
                    mean_arr = np.load(self.config['data_mean'])
                elif self.config['data_mean'].endswith('.binaryproto'):
                    mean_arr = load_binaryproto(self.config['data_mean'])
                else:
                    mean_arr = np.matrix(self.config['data_mean']).A1
                # print 'mean array shape: ', mean_arr.shape
                # print 'mean array: \n', mean_arr
            except:
                raise InitError('Failed to load "data_mean": ' +
                                str(self.config['data_mean']))

        if (int(self.config['mirror_trick']) not in [0, 1, 2]):
            raise InitError('"mirror_trick" must be one from [0,1,2]')

        print '\n===> CaffeFeatureExtractor.config: \n', self.config

        try:
            if(self.config['cpu_only']):
                caffe.set_mode_cpu()
            else:
                caffe.set_mode_gpu()
                caffe.set_device(int(self.config['gpu_id']))
        except Exception as err:
            raise InitError(
                'Exception from caffe.set_mode_xxx() or caffe.set_device(): ' + str(err))

        try:
            self.net=Classifier(self.config['network_prototxt'],
                                        self.config['network_caffemodel'],
                                        None,
                                        mean_arr,
                                        self.config['input_scale'],
                                        self.config['raw_scale'],
                                        self.config['channel_swap']
                                        )
        except Exception as err:
            raise InitError('Exception from Clssifier.__init__(): ' + str(err))

        if (self.config['feature_layer'] not in self.net.layer_dict.keys()):
            raise FeatureLayerError('Invalid feature layer name: '
                                        + self.config['feature_layer'])

#        self.net_blobs = OrderedDict([(k, v.data)
#                                  for k, v in self.net.blobs.items()])
#        print 'self.net_blobs: ', self.net_blobs
#        for k, v in self.net.blobs.items():
#            print k, v

        self.input_shape=self.net.blobs['data'].data.shape
        print '---> original input data shape (in prototxt): ', self.input_shape
        print '---> original batch_size (in prototxt): ', self.input_shape[0]

        self.batch_size=self.config['batch_size']
        print '---> batch size in the config: ', self.batch_size

        if self.config['mirror_trick'] > 0:
            print '---> need to double the batch size of the net input data because of mirror_trick'
            final_batch_size=self.batch_size * 2
        else:
            final_batch_size=self.batch_size

        print '---> will use a batch size: ', final_batch_size

        # reshape net into final_batch_size
        if self.input_shape[0] != final_batch_size:
            print '---> reshape net input batch size from %d to %d' % (self.input_shape[0], final_batch_size)
            self.net.blobs['data'].reshape(
                final_batch_size, self.input_shape[1], self.input_shape[2], self.input_shape[3])
            print '---> reshape the net blobs'
            self.net.reshape()

            self.input_shape=self.net.blobs['data'].data.shape

            print '---> after reshape, net input data shape: ', self.input_shape

        print '---> the final input data shape: ', self.input_shape

#        if self.config['mirror_trick'] > 0:
#            if self.batch_size < 2:
#                raise InitError('If using mirror_trick, batch_size of input "data" layer must > 1')
#
#            self.batch_size /= 2
# print 'halve the batch_size for mirror_trick eval: batch_size=',
# self.batch_size

    def __delete__(self):
        print 'delete CaffeFeatureExtractor object'

    def load_image(self, image_path):
        img=caffe.io.load_image(
            image_path, color=not self.config['image_as_grey'])
        if self.config['image_as_grey'] and img.shape[2] != 1:
            img = skimage.color.rgb2gray(img)
            img = img[:, :, np.newaxis]

        return img

    def load_images(self, image_path):
        pass

    def get_feature_layer_name(self, layer_name=None):
        if not layer_name:
            return self.config['feature_layer']

        if (layer_name in self.net.layer_dict.keys()):
            return layer_name
        else:
            raise FeatureLayerError('Invalid feature layer name: '
                                        + layer_name)
            return None

    def extract_feature(self, image, layer_name=None):
        layer_name = self.get_feature_layer_name(layer_name)

        feat_shp = self.net.blobs[layer_name].shape
        print 'feature layer shape: ', feat_shp

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
            img = image.astype(np.float32)  # data type must be float32

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
#        blobs = OrderedDict([(k, v.data)
#                             for k, v in self.net.blobs.items()])
#        print 'blobs: ', blobs
        feat_blob_data = self.net.blobs[layer_name].data

        if self.config['mirror_trick']:
            #            ftrs = blobs[layer_name][0:n_imgs * 2, ...]
            ftrs = feat_blob_data[0:n_imgs * 2, ...]
            if self.config['mirror_trick'] == 2:
                eltop_ftrs = np.maximum(ftrs[:n_imgs], ftrs[n_imgs:])
            else:
                eltop_ftrs = (ftrs[:n_imgs] + ftrs[n_imgs:]) * 0.5

            feature = eltop_ftrs[0]

        else:
            #            ftrs = blobs[layer_name][0:n_imgs, ...]
            ftrs = feat_blob_data[0:n_imgs, ...]
            feature = ftrs.copy()  # copy() is a must-have

        if cnt_load_img:
            print ('load %d images cost %f seconds, average time: %f seconds'
                   % (cnt_load_img, time_load_img, time_load_img / cnt_load_img))

        print ('predict %d images cost %f seconds, average time: %f seconds'
               % (cnt_predict, time_predict, time_predict / cnt_predict))

        feature = np.asarray(feature, dtype='float32')

        if self.config['normalize_output']:
            feat_norm = norm(feature)
            feature /= feat_norm

        return feature

    def extract_features_batch(self, images, layer_name=None):
        layer_name = self.get_feature_layer_name(layer_name)

        n_imgs = len(images)

        if (n_imgs > self.batch_size
                or (self.config['mirror_trick'] and n_imgs / 2 > self.batch_size)):
            raise ExtractionError('Number of input images > batch_size set in prototxt')

        feat_shp = self.net.blobs[layer_name].data.shape
        print 'feature layer shape: ', feat_shp

        features_shape = (len(images),) + feat_shp[1:]
        features = np.empty(features_shape, dtype='float32', order='C')
        print 'output features shape: ', features_shape

        # data type must be float32
        img_batch = [im.astype(np.float32) for im in images]

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
#        blobs = OrderedDict([(k, v.data)
#                             for k, v in self.net.blobs.items()])
#        print 'blobs: ', blobs
        feat_blob_data = self.net.blobs[layer_name].data

        if self.config['mirror_trick']:
            #            ftrs = blobs[layer_name][0:n_imgs * 2, ...]
            ftrs = feat_blob_data[0:n_imgs * 2, ...]
            if self.config['mirror_trick'] == 2:
                eltop_ftrs = np.maximum(ftrs[:n_imgs], ftrs[n_imgs:n_imgs * 2])
            else:
                eltop_ftrs = (ftrs[:n_imgs] + ftrs[n_imgs:n_imgs * 2]) * 0.5

            features = eltop_ftrs.copy()

        else:
            #            ftrs = blobs[layer_name][0:n_imgs, ...]
            ftrs = feat_blob_data[0:n_imgs, ...]
            features = ftrs.copy()  # copy() is a must-have

        print('Predict %d images, cost %f seconds, average time: %f seconds' %
              (cnt_predict, time_predict, time_predict / cnt_predict))

        features = np.asarray(features, dtype='float32')
        if self.config['normalize_output']:
            feat_norm = norm(features, axis=1)
            features = features / np.reshape(feat_norm, [-1, 1])

        return features

    def extract_features_for_image_list(self, image_list, img_root_dir=None, layer_name=None):
        layer_name = self.get_feature_layer_name(layer_name)

        feat_shp = self.net.blobs[layer_name].data.shape
        print 'feature layer shape: ', feat_shp

        features_shape = (len(image_list),) + feat_shp[1:]
        features = np.empty(features_shape, dtype='float32', order='C')
        print 'output features shape: ', features_shape
        img_batch = []

        cnt_load_img = 0
        time_load_img = 0.0
#        cnt_predict = 0
#        time_predict = 0.0

        for cnt, path in zip(range(features_shape[0]), image_list):
            t1 = time.clock()

            if img_root_dir:
                path = osp.join(img_root_dir, path)

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
        if self.config['normalize_output']:
            feat_norm = norm(features, axis=1)
            features = features / np.reshape(feat_norm, [-1, 1])

        return features


if __name__ == '__main__':
    def load_image_list(list_file_name):
        # list_file_path = os.path.join(img_dir, list_file_name)
        f = open(list_file_name, 'r')
        img_fn_list = []

        for line in f:
            if line.startswith('#'):
                continue

            items = line.split()
            img_fn_list.append(items[0].strip())

        f.close()

        return img_fn_list

    config_json = './extractor_config_sphere64.json'
    save_dir = 'feature_rlt_sphere64_noflip'

    image_dir = r'C:\zyf\github\mtcnn-caffe-good-new\face_aligner\face_chips'
    image_list_file = r'C:\zyf\github\lfw-evaluation-zyf\extract_face_features\face_chips\face_chips_list_2.txt'

    if not osp.exists(save_dir):
        os.makedirs(save_dir)

    # test extract_features_for_image_list()
    save_name = 'img_list_features.npy'

    img_list = load_image_list(image_list_file)

    print '\n===> test extract_features_for_image_list()'

    # init a feat_extractor, use a context to release caffe objects
    print '\n===> init a feat_extractor'
    feat_extractor = CaffeFeatureExtractor(config_json)

    ftrs = feat_extractor.extract_features_for_image_list(img_list, image_dir)
#    np.save(osp.join(save_dir, save_name), ftrs)

    root_len = len(image_dir)

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

        save_name = osp.splitext(base_name)[0] + '.npy'
        np.save(osp.join(save_sub_dir, save_name), ftrs[i])

    # test extract_feature()
#    print '\n===> test extract_feature()'
#    save_name_2 = 'single_feature.npy'
#    ftr = feat_extractor.extract_feature(osp.join(image_dir, img_list[0]))
#    np.save(osp.join(save_dir, save_name_2), ftr)
#
#    ft_diff = ftr - ftrs[0]
#    print 'ft_diff: ', ft_diff.sum()
