"""
Custom loss functions module
"""

import tensorflow as tf


class MyMSE(tf.keras.losses.Loss):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name='my_mse', **kwargs):
        super(MyMSE, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
        return loss
