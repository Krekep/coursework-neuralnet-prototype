"""
Custom loss functions module
"""
from abc import ABC

import tensorflow as tf
from tensorflow import keras


class RelativeError(tf.keras.losses.Loss, ABC):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name='relative', **kwargs):
        super(RelativeError, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        y_upd = tf.where(y_true == 0.0, 1.0, y_true)
        y = tf.math.divide(y_pred, y_upd)
        loss = tf.math.reduce_mean(tf.abs(y - 1))
        return loss


class RelativeAbsoluteError(tf.keras.losses.Loss, ABC):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name='rae', **kwargs):
        super(RelativeAbsoluteError, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        true_mean = tf.reduce_mean(y_true)
        squared_error_num = tf.reduce_sum(tf.abs(y_true - y_pred))
        squared_error_den = tf.reduce_sum(tf.abs(y_true - true_mean))

        if squared_error_den == 0:
            squared_error_den = 1

        loss = squared_error_num / squared_error_den
        return loss


class MaxAbsoluteDeviation(tf.keras.losses.Loss, ABC):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name='my_mae', **kwargs):
        super(MaxAbsoluteDeviation, self).__init__(reduction=reduction, name=name, **kwargs)

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss = tf.math.reduce_max(tf.math.abs(y_true - y_pred))
        return loss


# Non-differentiable
class InlierRatio(tf.keras.losses.Loss, ABC):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name='inlier_ratio', treeshold=0.05, **kwargs):
        super(InlierRatio, self).__init__(reduction=reduction, name=name, **kwargs)
        self.treeshold = treeshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0.0, 1.0, y_true))
        loss = tf.math.reduce_mean(tf.where(tf.abs(y) <= self.treeshold, 0.0, 1.0))
        return loss


class MaxDeviation(tf.keras.losses.Loss, ABC):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name='max_deviation', treeshold=0.05, **kwargs):
        super(MaxDeviation, self).__init__(reduction=reduction, name=name, **kwargs)
        self.treeshold = treeshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0, 1, y_true))
        loss = tf.math.reduce_max(tf.where(tf.abs(y) <= self.treeshold, 0.0, abs(y)))
        return loss


class MeanDeviation(tf.keras.losses.Loss, ABC):
    def __init__(self, reduction=tf.keras.losses.Reduction.NONE,
                 name='mean_deviation', treeshold=0.05, **kwargs):
        super(MeanDeviation, self).__init__(reduction=reduction, name=name, **kwargs)
        self.treeshold = treeshold

    def __call__(self, y_true, y_pred, sample_weight=None):
        y = tf.math.divide((y_true - y_pred), tf.where(y_true == 0, 1, y_true))
        loss = tf.math.reduce_mean(tf.where(tf.abs(y) <= self.treeshold, self.treeshold, abs(y)))
        return loss


# Reduction should be set to None?
_losses: dict = {
    "BinaryCrossentropy": keras.losses.BinaryCrossentropy(),                        # Not usable in my task
    "BinaryFocalCrossentropy": keras.losses.BinaryFocalCrossentropy(),              # Not usable in my task
    "CategoricalCrossentropy": keras.losses.CategoricalCrossentropy(),              # Not usable in my task
    "CategoricalHinge": keras.losses.CategoricalHinge(),                            # Not usable in my task
    "CosineSimilarity": keras.losses.CosineSimilarity(),                            # Not usable in my task
    "Hinge": keras.losses.Hinge(),                                                  # Not usable in my task
    "SparseCategoricalCrossentropy": keras.losses.SparseCategoricalCrossentropy(),  # Not usable in my task
    "SquaredHinge": keras.losses.SquaredHinge(),                                    # Not usable in my task

    "Huber": keras.losses.Huber(),
    "KLDivergence": keras.losses.KLDivergence(),
    "LogCosh": keras.losses.LogCosh(),
    "MeanAbsoluteError": keras.losses.MeanAbsoluteError(),
    "MeanAbsolutePercentageError": keras.losses.MeanAbsolutePercentageError(),
    "MeanSquaredError": keras.losses.MeanSquaredError(),
    "MeanSquaredLogarithmicError": keras.losses.MeanSquaredLogarithmicError(),
    "Poisson": keras.losses.Poisson(),
    "RelativeError": RelativeError(),
    "RelativeAbsoluteError": RelativeAbsoluteError(),
    "MaxAbsoluteDeviation": MaxAbsoluteDeviation(),
}

_metrics: dict = {
    "AUC": keras.metrics.AUC(),                                                         # Not usable in my task
    "Accuracy": keras.metrics.Accuracy(),                                               # Not usable in my task
    "BinaryAccuracy": keras.metrics.BinaryAccuracy(),                                   # Not usable in my task
    "BinaryIoU": keras.metrics.BinaryIoU(),                                             # Not usable in my task
    "FalseNegatives": keras.metrics.FalseNegatives(),                                   # Not usable in my task
    "FalsePositives": keras.metrics.FalsePositives(),                                   # Not usable in my task
    "IoU": keras.metrics.IoU(),                                                         # Not usable in my task
    "MeanIoU": keras.metrics.MeanIoU(),                                                 # Not usable in my task
    "OneHotIoU": keras.metrics.OneHotIoU(),                                             # Not usable in my task
    "OneHotMeanIoU": keras.metrics.OneHotMeanIoU(),                                     # Not usable in my task
    "Precision": keras.metrics.Precision(),                                             # Not usable in my task
    "PrecisionAtRecall": keras.metrics.PrecisionAtRecall(),                             # Not usable in my task
    "Recall": keras.metrics.Recall(),                                                   # Not usable in my task
    "RecallAtPrecision": keras.metrics.RecallAtPrecision(),                             # Not usable in my task
    "SensitivityAtSpecificity": keras.metrics.SensitivityAtSpecificity(),               # Not usable in my task
    "SparseCategoricalAccuracy": keras.metrics.SparseCategoricalAccuracy(),             # Not usable in my task
    "SparseTopKCategoricalAccuracy": keras.metrics.SparseTopKCategoricalAccuracy(),     # Not usable in my task
    "SpecificityAtSensitivity": keras.metrics.SpecificityAtSensitivity(),               # Not usable in my task
    "Sum": keras.metrics.Sum(),                                                         # Not usable in my task
    "TopKCategoricalAccuracy": keras.metrics.TopKCategoricalAccuracy(),                 # Not usable in my task
    "TrueNegatives": keras.metrics.TrueNegatives(),                                     # Not usable in my task
    "TruePositives": keras.metrics.TruePositives(),                                     # Not usable in my task

    "Mean": keras.metrics.Mean(),                                                       # Not usable in my task?
    "MeanRelativeError": keras.metrics.MeanRelativeError(),                             # Not usable in my task?
    "MeanTensor": keras.metrics.MeanTensor(),                                           # Not usable in my task?

    "RootMeanSquaredError": keras.metrics.RootMeanSquaredError(),
    "InlierRatio": InlierRatio(),
    "MaxDeviation": MaxDeviation(),
    "MeanDeviation": MeanDeviation(),
}

_metrics = dict(_losses, **_metrics)


def get_metric(name: str):
    return _metrics.get(name)


def get_losses(name: str):
    return _losses.get(name)
