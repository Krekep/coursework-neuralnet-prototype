from collections import defaultdict

from tensorflow import Tensor
import keras.activations, keras.initializers
import numpy as np

from networks.layers.dense import MyDense


def _check_dimension(x) -> tuple:
    """
    Check that x have at least 2 dimension and complement otherwise

    :param x:
    :return:
    """

    if isinstance(x, list):
        if len(x) != 0:
            if isinstance(x[0], list):
                return True, np.array(x)
            else:
                return True, np.array([x])
        return False, None
    if isinstance(x, np.ndarray):
        if x.ndim == 1:
            return True, x.reshape((x.shape[0], 1))
        return x.ndim >= 2, x
    if isinstance(x, Tensor):
        x = x.numpy()
        if x.ndim == 1:
            return True, x.reshape((x.shape[0], 1))
        return x.ndim >= 2, x
    return False, None


class ILayer:
    """
    Interface class for working with layers.

    """

    def __init__(self, inp_size, shape, activation, weight, bias, layer_type="Dense", name="net", is_debug=False, **kwargs):
        try:
            self.layer = _create_functions[layer_type](inp_size, shape, activation, weight, bias, **kwargs)

            self._name = name
            self._shape = shape
            self._input_size = inp_size
            self._is_debug = is_debug
            self.set_name(name)
        except Exception as e:
            print(e)
            self.layer = None

    @classmethod
    def create_dense(cls,
                     inp_size,
                     shape,
                     activation=keras.activations.linear,
                     weight=keras.initializers.get('ones'),
                     bias=keras.initializers.get('zeros'),
                     **kwargs
                     ):
        if kwargs.get("layer_type") is not None:
            ilayer = cls(inp_size, shape, activation, weight, bias, **kwargs)
        else:
            ilayer = cls(inp_size, shape, activation, weight, bias, layer_type="Dense", **kwargs)
        return ilayer

    def get_config(self) -> dict:
        return self.layer.get_config()

    def get_weights(self):
        return self.layer.get_weights()

    def to_dict(self):
        return self.layer.to_dict()

    @classmethod
    def from_dict(cls, config):
        res = cls.create_dense(inp_size=config["inp_size"], shape=config["shape"], layer_type=config["layer_type"])
        res.layer.from_dict(config)

        return res

    def set_name(self, name: str) -> None:
        """
        Set layer name

        Parameters
        ----------
        name: str
            New name
        Returns
        -------
        None
        """
        self._name = name

    @property
    def get_name(self) -> str:
        return self._name

    @property
    def get_shape(self) -> int:
        """
        Get count of elements for current layer

        Returns
        -------
        shape: int
            Layer shape
        """

        return self._shape

    @property
    def get_input_size(self) -> int:
        """
        Get input vector size for current layer

        Returns
        -------
        size: int
            Input vector size
        """

        return self._input_size

    def __str__(self) -> str:
        """
        Get a string representation of the layer

        Returns
        -------
        result: str
        """

        return str(self.layer)

    def __call__(self, x, **kwargs):
        is_suitable, x = _check_dimension(x)
        if not is_suitable:
            raise Exception("Input vector for layer must have at least 2 dimension")
        return self.layer(x)


_create_functions = defaultdict(lambda: MyDense)
_create_functions["Dense"] = MyDense

