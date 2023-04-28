"""
Custom loss functions module
"""
from abc import ABC
from typing import Callable

import tensorflow as tf
from tensorflow import keras


class RelativeError(tf.keras.losses.Loss, ABC):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="relative", **kwargs
    ):
        super(RelativeError, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_upd = tf.where(y_true == 0.0, 1.0, y_true)
        y = tf.math.divide(y_pred, y_upd)
        loss = tf.math.reduce_mean(tf.abs(y - 1))
        return loss


class RelativeAbsoluteError(tf.keras.losses.Loss, ABC):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE, name="rae", **kwargs):
        super(RelativeAbsoluteError, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        true_mean = tf.reduce_mean(y_true)
        squared_error_num = tf.reduce_sum(tf.abs(y_true - y_pred))
        squared_error_den = tf.reduce_sum(tf.abs(y_true - true_mean))

        squared_error_den = tf.cond(
            pred=tf.equal(squared_error_den, tf.constant(0.0)),
            true_fn=lambda: tf.constant(1.0),
            false_fn=lambda: squared_error_den,
        )

        loss = squared_error_num / squared_error_den
        return loss


class MaxAbsoluteDeviation(tf.keras.losses.Loss, ABC):
    def __init__(
        self, reduction=tf.keras.losses.Reduction.NONE, name="my_mae", **kwargs
    ):
        super(MaxAbsoluteDeviation, self).__init__(
            reduction=reduction, name=name, **kwargs
        )

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs(y_true - y_pred))
        return loss


# Reduction should be set to None?
_losses: dict = {
    # "BinaryCrossentropy": keras.losses.BinaryCrossentropy(),                        # Not usable in my task
    # "BinaryFocalCrossentropy": keras.losses.BinaryFocalCrossentropy(),              # Not usable in my task
    # "CategoricalCrossentropy": keras.losses.CategoricalCrossentropy(),              # Not usable in my task
    # "CategoricalHinge": keras.losses.CategoricalHinge(),                            # Not usable in my task
    # "CosineSimilarity": keras.losses.CosineSimilarity(),                            # Not usable in my task
    # "Hinge": keras.losses.Hinge(),                                                  # Not usable in my task
    # "SparseCategoricalCrossentropy": keras.losses.SparseCategoricalCrossentropy(),  # Not usable in my task
    # "SquaredHinge": keras.losses.SquaredHinge(),                                    # Not usable in my task
    "Huber": keras.losses.Huber(),
    "KLDivergence": keras.losses.KLDivergence(),  # If you're using `model.compile()`, did you forget to provide a `loss`argument?
    "LogCosh": keras.losses.LogCosh(),  # If you're using `model.compile()`, did you forget to provide a `loss`argument?
    "MeanAbsoluteError": keras.losses.MeanAbsoluteError(),  # If you're using `model.compile()`, did you forget to provide a `loss`argument?
    "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError(),
    "MeanSquaredError": keras.losses.MeanSquaredError(),
    "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError(),  # If you're using `model.compile()`, did you forget to provide a `loss`argument?
    "Poisson": keras.losses.Poisson(),
    "RelativeError": RelativeError(),
    "RelativeAbsoluteError": RelativeAbsoluteError(),
    "MaxAbsoluteDeviation": MaxAbsoluteDeviation(),
}


def get_loss(name: str):
    return _losses.get(name)


def get_all_loss_functions() -> dict[str, Callable]:
    return _losses
