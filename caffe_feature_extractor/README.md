# Caffe Feature Extractor
A wrapper for extractring features from Caffe network, with a config file to define network parameter.

## Python requirements:
```
pycaffe
numpy
skimage
json
```

## extractor config example

1. If using caffe.io.load_image (=skimage.io.imread) to load images, then the config file might looks like:

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

Note: 

 *'batch_size'* in the config json file would overwrite the 'batch size' in the prototxt;

 *'mirror_trick'*: =0, original features; =1, eltsum(original, mirrored)/2; =2, eltmax(original, mirrored);

 *'normalize_output'*: =1, will do L2-normalization before output; =0, no normalization.

2. If using opencv (cv2.imread) to load images, then the config file might looks like:
 
```json
{
    "network_prototxt": "path/to/prototxt",
    "network_caffemodel": "path/to/caffemodel",
    "data_mean": "path/to/meanfile",
    "feature_layer": "fc5",
    "batch_size": 10,
    "input_scale": 0.0078125,
    "raw_scale": 1.0,
    "channel_swap": "0, 1, 2",
    "mirror_trick": 1,
    "image_as_grey": 0,
    "normalize_output": 1
}
```