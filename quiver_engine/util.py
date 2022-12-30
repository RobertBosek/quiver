from __future__ import absolute_import, division, print_function
import json
from flask.json import jsonify
import numpy as np
import keras.utils as image
import keras.backend as K
from contextlib import contextmanager
from quiver_engine.imagenet_utils import preprocess_input, decode_imagenet_predictions
from os import path


def validate_launch(html_base_dir):
    print('Starting webserver from:', html_base_dir)
    assert path.exists(path.join(html_base_dir, 'quiverboard')), 'Quiverboard must be a ' \
                                                                       'subdirectory of {}'.format(html_base_dir)
    assert path.exists(path.join(html_base_dir, 'quiverboard', 'dist')), 'Dist must be a ' \
                                                                               'subdirectory of quiverboard'
    assert path.exists(
        path.join(html_base_dir, 'quiverboard', 'dist', 'index.html')), 'Index.html missing'

def get_evaluation_context():
    return get_evaluation_context_getter()()

def get_evaluation_context_getter():
    if K.backend() == 'tensorflow':
        import tensorflow as tf
        return tf.compat.v1.get_default_graph().as_default

    if K.backend() == 'theano':
        return contextmanager(lambda: (yield))

def get_input_config(model):
    '''
        returns a tuple (inputDimensions, numChannels)
    '''
    print(K.image_data_format())
    print(model.get_layer(index=0).get_config())
    print((
        model.get_layer(index=0).get_config()["batch_input_shape"][1:3],
        model.get_layer(index=0).get_config()["batch_input_shape"][3]
    ))
    return (
        model.get_layer(index=0).get_config()["batch_input_shape"][1:3],
        model.get_layer(index=0).get_config()["batch_input_shape"][3]
    )
    return (
        model.get_layer(index=0).get_config()["batch_input_shape"][0:3],
        model.get_layer(index=0).get_config()["batch_input_shape"][3]
    ) if K.image_data_format() == 'th' else (
        #tf ordering
	    model.get_layer(index=0).get_config()["batch_input_shape"][0:3],
        model.get_layer(index=0).get_config()["batch_input_shape"][3]
    )

def decode_predictions(preds, classes, top):
    if not classes:
        print("Warning! you didn't pass your own set of classes for the model therefore imagenet classes are used")
        return decode_imagenet_predictions(preds, top)

    if len(preds.shape) != 2 or preds.shape[1] != len(classes):
        raise ValueError('you need to provide same number of classes as model prediction output ' + \
                         'model returns %s predictions, while there are %s classes' % (
                             preds.shape[1], len(classes)))
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [("", classes[i], pred[i]) for i in top_indices]
        results.append(result)
    return results

# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    return x

def load_img_scaled(input_path, target_shape, grayscale=False):
    return np.expand_dims(
        image.img_to_array(image.load_img(input_path, target_size=target_shape, grayscale=grayscale)) / 255.0,
        axis=0
    )

def load_img(input_path, target_shape, grayscale=False, mean=None, std=None):
    print(input_path, target_shape)
    img = image.load_img(input_path, target_size=target_shape,
                         grayscale=grayscale)
    img_arr = np.expand_dims(image.img_to_array(img), axis=0)
    if not grayscale:
        img_arr = preprocess_input(img_arr, mean=mean, std=std)
    return img_arr


def get_jsonable_obj(obj):
    return json.loads(get_json(obj))

def get_json(obj):
    return json.dumps(obj, default=get_json_type)

def safe_jsonify(obj):
    return jsonify(get_jsonable_obj(obj))

def get_json_type(obj):

    # if obj is any numpy type
    if type(obj).__module__ == np.__name__:
        return obj.item()

    # if obj is a python 'type'
    if type(obj).__name__ == type.__name__:
        return obj.__name__

    raise TypeError('Not JSON Serializable')
