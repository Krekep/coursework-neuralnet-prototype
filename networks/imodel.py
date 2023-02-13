from collections import defaultdict
from typing import List
import json

import keras
import keras.activations
import keras.initializers
import numpy as np
import tensorflow as tf

from networks import activations
from networks.densenet import DenseNet
from networks.layers.dense import MyDense


def _get_act_and_init(kwargs: dict, default_act, default_out_act, default_init):
    if kwargs.get("activation") is None:
        activation = default_act
    else:
        activation = kwargs["activation"]
        kwargs.pop("activation")

    if kwargs.get("out_activation") is None:
        out_activation = default_out_act
    else:
        out_activation = kwargs["out_activation"]
        kwargs.pop("out_activation")

    if kwargs.get("weight") is None:
        weight = default_init
    else:
        weight = kwargs["weight"]
        kwargs.pop("weight")

    if kwargs.get("biases") is None:
        biases = default_init
    else:
        biases = kwargs["biases"]
        kwargs.pop("biases")

    return activation, out_activation, weight, biases, kwargs


class IModel(object):
    """
    Interface class for working with neural networks
    """

    def __init__(self, input_size, block_size, output_size,
                 activation_func=keras.activations.sigmoid,
                 out_activation=tf.keras.activations.linear,
                 weight_init=tf.random_normal_initializer(),
                 bias_init=tf.random_normal_initializer(),
                 normalization=1.0,
                 name="net",
                 net_type='DenseNet',
                 is_debug=False, **kwargs):

        try:
            self.network = _create_functions[net_type](input_size, block_size, activation_func=activation_func,
                                                       out_activation=out_activation,
                                                       weight=weight_init, biases=bias_init, output_size=output_size,
                                                       is_debug=is_debug, **kwargs)

            # self.network = DenseNet(input_size, block_size, activation_func=activation_func,
            #                         out_activation=out_activation,
            #                         weight=weight_init, biases=bias_init, output_size=output_size,
            #                         is_debug=is_debug, **kwargs)
            self._input_size = input_size
            self._output_size = output_size
            self._normalization = normalization
            self._shape = block_size
            self._train_history = None
            self._name = name
            self._is_debug = is_debug
            self.set_name(name)

        except Exception as e:
            print(e)
            self.network = None

    def compile(self, rate=1e-2, optimizer=tf.keras.optimizers.SGD,
                loss_func=tf.keras.losses.MeanSquaredError(), metrics=None):
        if metrics is None:
            metrics = [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(),
                       tf.keras.metrics.MeanSquaredLogarithmicError()]

        self.network.compile(optimizer=optimizer(learning_rate=rate), loss=loss_func, metrics=metrics,
                             run_eagerly=False)

    def feedforward(self, inputs: np.ndarray) -> tf.Tensor:
        """
        Return network answer for passed input

        Parameters
        ----------
        inputs: np.ndarray
            Input activation vector
        Returns
        -------
        outputs: tf.Tensor
            Network answer
        """

        return tf.math.scalar_mul(self._normalization, self.network(inputs))

    def train(self, x_data: np.ndarray, y_data: np.ndarray, validation_split=0.0,
              epochs=50, mini_batch_size=None, callbacks=None,
              verbose='auto') -> keras.callbacks.History:
        """
        Train network on passed dataset and return training history

        Parameters
        ----------
        x_data: np.ndarray
            Array of input vectors
        y_data: np.ndarray
            Array of output vectors
        validation_split: float
            Percentage of data to validate
        epochs: int
            Count of epochs for training
        mini_batch_size: int
            Size of batches
        callbacks: list
            List of tensorflow callbacks for fit function
        verbose: int
            Output accompanying training

        Returns
        -------
        history: tf.keras.callbacks.History
            History of training
        """
        if not self._is_debug:
            callbacks = []
        self._train_history = self.network.fit(x_data, y_data, batch_size=mini_batch_size,
                                               callbacks=callbacks,
                                               validation_split=validation_split, epochs=epochs, verbose=verbose)
        return self._train_history

    # def save_weights(self, path, **kwargs):
    #     with open(path, 'r') as f:
    #         s = self.network.to_json(**kwargs)
    #         f.write(s)
    #
    # def load_weights(self, path, **kwargs):
    #     with open(path, "w") as f:
    #         s = f.readline()
    #         self.network = keras.models.model_from_json(s, **kwargs)

    def to_dict(self, path, **kwargs):
        config = self.network.to_dict(**kwargs)
        with open(path, "w") as f:
            f.write(json.dumps(config))

    def from_dict(self, path, **kwargs):
        with open(path, "r") as f:
            config = json.loads(f.readline())
            self.network.from_dict(config)
            self.set_name(config["name"])

    def save_model(self, path, **kwargs):
        self.network.save(path, **kwargs)

    def set_name(self, name: str) -> None:
        """
        Set network name

        Parameters
        ----------
        name: str
            New name
        Returns
        -------
        None
        """
        self.network.set_name(name)
        self._name = name

    @property
    def get_name(self) -> str:
        return self._name

    @property
    def get_shape(self) -> List[int]:
        """
        Get shape for current network

        Returns
        -------
        shape: List[int]
            Network shape
        """

        return self._shape

    @property
    def get_input_size(self) -> int:
        """
        Get input vector size for current network

        Returns
        -------
        size: int
            Input vector size
        """

        return self._input_size

    @property
    def get_output_size(self) -> int:
        """
        Get output vector size for current network

        Returns
        -------
        size: int
            Output vector size
        """

        return self._output_size

    def __str__(self) -> str:
        """
        Get a string representation of the neural network

        Returns
        -------
        result: str
        """

        return str(self.network)

    @classmethod
    def load_model(cls, path, **kwargs):
        res = cls(1, [1], 1)
        res.network = keras.models.load_model(path, custom_objects={'Dense': MyDense()}, **kwargs)

        return res

    @classmethod
    def create_neuron(cls, input_size, output_size, shape, **kwargs):
        activation, out_activation, weight, biases, kwargs = _get_act_and_init(kwargs,
                                                                               keras.activations.sigmoid,
                                                                               keras.activations.linear,
                                                                               tf.random_normal_initializer())

        try:
            res = cls(input_size=input_size,
                      block_size=shape,
                      output_size=output_size,
                      activation_func=activation,
                      out_activation=out_activation,
                      bias_init=biases,
                      weight_init=weight,
                      **kwargs)
        except Exception as e:
            print(e)
            res = None

        return res

    @classmethod
    def create_perceptron(cls, input_size, output_size, shape, threshold=1, **kwargs):

        activation, out_act, weight, biases, kwargs = _get_act_and_init(kwargs,
                                                                        activations.perceptron_threshold(threshold),
                                                                        activations.perceptron_threshold(threshold),
                                                                        tf.random_normal_initializer())

        try:
            res = cls(input_size=input_size,
                      block_size=shape,
                      output_size=output_size,
                      activation_func=activation,
                      out_activation=out_act,
                      bias_init=biases,
                      weight_init=weight,
                      **kwargs)
        except Exception as e:
            print(e)
            res = None

        return res


_create_functions = defaultdict(lambda: DenseNet)
_create_functions["DenseNet"] = DenseNet
