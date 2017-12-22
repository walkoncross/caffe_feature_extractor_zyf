# Caffe Feature Extractor
A wrapper for extractring features from Caffe network, with a config file to define network parameter.

## Contents
```
---
|- caffe_feature_extractor  # caffe feature extractor
|- face_feature_extractor   # detect faces using MTCNN, align faces and extract features
|- utils                    # utils for compare similarity between two features
---
```

## Python requirements:
```
pycaffe
numpy
skimage
json
```

## extractor config example
```json
{
    "network_prototxt": "path/to/prototxt",
    "network_caffemodel": "path/to/caffemodel",
    "data_mean": "path/to/meanfile",
    "feature_layer": "fc5",
    "batch_size": 10,
    "input_scale": 0.0078125,
    "raw_scale": 255.0,
    "channel_swap": "2, 1, 0",
    "mirror_trick": 1,
    "image_as_grey": 0,
    "normalize_output": 1
}
```