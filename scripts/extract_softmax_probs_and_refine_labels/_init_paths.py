import sys
import os.path as osp

sys.path.append(osp.join(osp.dirname(__file__), '../..'))

# set caffe path
caffe_root = '/opt/caffe/'
sys.path.insert(0, caffe_root + 'python')