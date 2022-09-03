import numpy as np
import tensorflow as tf
import keras

from project.networks import densenet
from project.networks.densenet import DenseNet

from project.networks import losses
from project.networks.losses import MyMSE


class INetwork(object):
    """
    Interface class for working with neural networks
    """

    def __init__(self, input_size=2, block_size=None, output_size=10, rate=1e-2, is_debug=False, **kwargs):
        self.DenseNetwork = DenseNet(input_size=input_size, block_size=block_size, output_size=output_size,
                                     is_debug=is_debug, **kwargs)
        self.DenseNetwork.compile(optimizer="SGD", loss=MyMSE(), run_eagerly=False)
        self._input_size = input_size
        self._output_size = output_size

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

        return self.DenseNetwork(inputs)

    def train(self, x_data: np.ndarray, y_data: np.ndarray, validation_split=0.0,
              epochs=50, mini_batch_size=None,
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
        verbose: int
            Output accompanying training

        Returns
        -------
        history: tf.keras.callbacks.History
            History of training
        """

        return self.DenseNetwork.fit(x_data, y_data, batch_size=mini_batch_size, validation_split=validation_split,
                                     epochs=epochs, verbose=verbose)

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
        self.DenseNetwork.set_name(name)
        # self.name = name

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

        return str(self.DenseNetwork)
