from collections import defaultdict

from tensorflow import Tensor
import keras.activations, keras.initializers
import numpy as np

from networks import activations
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
            return True, x.reshape((1, x.shape[0]))
        return x.ndim >= 2, x
    if isinstance(x, Tensor):
        x = x.numpy()
        if x.ndim == 1:
            return True, x.reshape((1, x.shape[0]))
        return x.ndim >= 2, x
    return False, None


def create(inp_size,
           shape,
           activation=activations.get("linear"),
           weight=keras.initializers.get("ones"),
           bias=keras.initializers.get("zeros"),
           layer_type="Dense",
           is_debug=False,
           **kwargs):
    layer = _create_functions[layer_type](
        inp_size, shape, activation, weight, bias, is_debug=is_debug, **kwargs
    )
    return layer


def create_dense(
        inp_size,
        shape,
        activation=keras.activations.linear,
        weight=keras.initializers.get("ones"),
        bias=keras.initializers.get("zeros"),
        **kwargs
):
    layer = create(
        inp_size, shape, activation, weight, bias, layer_type="Dense", **kwargs
    )
    return layer


def from_dict(config):
    res = create(
        inp_size=config["inp_size"],
        shape=config["shape"],
        layer_type=config["layer_type"],
    )
    res.from_dict(config)

    return res


_create_functions = defaultdict(lambda: MyDense)
_create_functions["Dense"] = MyDense
