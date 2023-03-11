from typing import Optional, List, Tuple, Dict

import tensorflow as tf
from tensorflow import keras
from keras import layers

from networks import activations
from networks.config_format import LAYER_DICT_NAMES


def _dec_params_to_list(
    params: Optional[Dict[str, float]]
) -> Optional[List[Tuple[str, float]]]:
    if params is None:
        return None
    res = []
    for key in params:
        res.append((key, params[key]))
    return res


def _dec_params_from_list(
    params: Optional[List[Tuple[str, float]]]
) -> Optional[Dict[str, float]]:
    if params is None:
        return None
    res = {}
    for pair in params:
        res[pair[0]] = pair[1]
    return res


class MyDense(keras.layers.Layer):
    def __init__(
        self,
        input_dim=32,
        units=32,
        activation_func=tf.keras.activations.linear,
        weight_initializer=tf.random_normal_initializer(),
        bias_initializer=tf.random_normal_initializer(),
        is_debug=False,
        **kwargs,
    ):
        activation_names = None
        decorator_params = None
        if "activation_names" in kwargs.keys():
            activation_names = kwargs.get("activation_names")
            kwargs.pop("activation_names")

        if "decorator_params" in kwargs.keys():
            decorator_params = kwargs.get("decorator_params")
            kwargs.pop("decorator_params")

        if not isinstance(decorator_params, dict) and decorator_params is not None:
            raise "Additional parameters for activation function must be dictionary"

        if input_dim == 0 or units == 0:
            raise "Layer cannot have zero inputs or zero size"

        super(MyDense, self).__init__(**kwargs)
        w_init = weight_initializer
        self.w = tf.Variable(
            initial_value=w_init(shape=(input_dim, units), dtype="float32"), name=f"Var_w_{self.name}",
            trainable=True,
        )
        b_init = bias_initializer
        self.b = tf.Variable(
            initial_value=b_init(shape=(units,), dtype="float32"), name=f"Var_w_{self.name}", trainable=True
        )

        self.units = units
        self.input_dim = input_dim
        self._is_debug = is_debug
        self.activation_func = activation_func
        self.weight_initializer = weight_initializer
        self.bias_initializer = bias_initializer
        self.activation_name = activation_names
        self.decorator_params: Optional[dict] = decorator_params

    def call(self, inputs, **kwargs):
        f = tf.matmul(inputs, self.w)
        s = f + self.b
        if self.decorator_params is None:
            return self.activation_func(s)
        else:
            return self.activation_func(s, **self.decorator_params)

    def __str__(self):
        res = f"Layer {self.name}\n"
        res += f"weights shape = {self.w.shape}\n"
        if self._is_debug:
            # res += f"weights = {self.w.numpy()}\n"
            # res += f"biases = {self.b.numpy()}\n"
            res += f"activation = {self.activation_name}\n"
        return res

    def get_config(self):
        config = super(MyDense, self).get_config()  # get config of the base Layer class

        config.update(
            {
                "units": self.units,
                "activation_func": tf.keras.activations.serialize(self.activation_func),
                "weight": tf.keras.initializers.serialize(self.weight_initializer),
                "biases": tf.keras.initializers.serialize(self.bias_initializer),
            }
        )
        # you need to serialise the callable activation function

        return config

    def to_dict(self):
        w = self.w.value().numpy()
        b = self.b.value().numpy()
        res = {
            LAYER_DICT_NAMES["shape"]: self.units,
            LAYER_DICT_NAMES["inp_size"]: self.input_dim,
            LAYER_DICT_NAMES["weights"]: w.tolist(),
            LAYER_DICT_NAMES["biases"]: b.tolist(),
            LAYER_DICT_NAMES["layer_type"]: type(self).__name__,
            LAYER_DICT_NAMES["dtype"]: w.dtype.name,
            LAYER_DICT_NAMES["activation"]: self.activation_func.__name__
            if self.activation_name is None
            else self.activation_name,
            LAYER_DICT_NAMES["decorator_params"]: _dec_params_to_list(
                self.decorator_params
            ),
        }

        return res

    def from_dict(self, config):
        w = config[LAYER_DICT_NAMES["weights"]]
        b = config[LAYER_DICT_NAMES["biases"]]
        act = config[LAYER_DICT_NAMES["activation"]]
        dec_params = _dec_params_from_list(config[LAYER_DICT_NAMES["decorator_params"]])
        self.w = tf.Variable(
            initial_value=w,
            dtype=config[LAYER_DICT_NAMES["dtype"]],
            trainable=True,
        )
        self.b = tf.Variable(
            initial_value=b, dtype=config[LAYER_DICT_NAMES["dtype"]], trainable=True
        )
        self.activation_func = activations.deserialize(act)
        self.activation_name = act
        self.decorator_params = dec_params

    @classmethod
    def from_config(cls, config):
        return cls(**config)
