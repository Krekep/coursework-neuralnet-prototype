from typing import Callable

import tensorflow as tf


def perceptron_threshold(threshold):
    def activation(x):
        return tf.where(x >= threshold, 1.0, 0.0)
    return activation


def serialize(func: Callable):
    # name = _activation_serialize.get(func.__name__)
    return func.__name__


def deserialize(name: str, is_decorator: bool = False, **kwargs):
    res = _activation_name[name]
    if is_decorator:
        res = res(**kwargs)

    return res


_decorated_activation = {'perceptron_threshold'}


_activation_name = {
    'perceptron_threshold': perceptron_threshold,
    'elu': tf.keras.activations.elu,
    'relu': tf.keras.activations.relu,
    'gelu': tf.keras.activations.gelu,
    'selu': tf.keras.activations.selu,
    'exponential': tf.keras.activations.exponential,
    'linear': tf.keras.activations.linear,
    'sigmoid': tf.keras.activations.sigmoid,
    'hard_sigmoid': tf.keras.activations.hard_sigmoid,
    'swish': tf.keras.activations.swish,
    'tanh': tf.keras.activations.tanh,
    'softplus': tf.keras.activations.softplus,
    'softmax': tf.keras.activations.softmax,
    'softsign': tf.keras.activations.softsign,
}

_activation_serialize = {v: k for k, v in _activation_name.items()}
