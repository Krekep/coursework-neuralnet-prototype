import os
from collections import defaultdict
from typing import List, Callable, Optional, Dict
import json

import keras
import keras.activations
import keras.initializers
import numpy as np
import tensorflow as tf

from networks import activations, cpp_utils
from networks.config_format import LAYER_DICT_NAMES, HEADER_OF_FILE
from networks.topology.densenet import DenseNet


def _get_act_and_init(
    kwargs: dict,
    default_act,
    default_dec: Optional[List[Optional[Dict[str, float]]]],
    default_init,
):
    if kwargs.get("activation") is None:
        activation = default_act
    else:
        activation = kwargs["activation"]
        kwargs.pop("activation")

    if kwargs.get("decorator_params") is None:
        decorator_params = default_dec
    else:
        decorator_params = kwargs["decorator_params"]
        kwargs.pop("decorator_params")

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

    return activation, decorator_params, weight, biases, kwargs


class IModel(object):
    """
    Interface class for working with neural topology
    """

    def __init__(
        self,
        input_size: int,
        block_size: List[int],
        output_size: int,
        activation_func=keras.activations.sigmoid,
        weight_init=tf.random_normal_initializer(),
        bias_init=tf.random_normal_initializer(),
        name="net",
        net_type="DenseNet",
        is_debug=False,
        **kwargs,
    ):
        self.network = _create_functions[net_type](
            input_size,
            block_size,
            activation_func=activation_func,
            weight=weight_init,
            biases=bias_init,
            output_size=output_size,
            is_debug=is_debug,
            **kwargs,
        )

        # self.network = DenseNet(input_size, block_size, activation_func=activation_func,
        #                         out_activation=out_activation,
        #                         weight=weight_init, biases=bias_init, output_size=output_size,
        #                         is_debug=is_debug, **kwargs)
        self._input_size = input_size
        self._output_size = output_size
        self._shape = block_size
        self._train_history = None
        self._name = name
        self._is_debug = is_debug
        self.set_name(name)

    def compile(
        self,
        rate=1e-2,
        optimizer=tf.keras.optimizers.SGD,
        loss_func=tf.keras.losses.MeanSquaredError(),
        metrics=None,
    ):
        if metrics is None:
            metrics = [
                tf.keras.metrics.MeanSquaredError(),
                tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.MeanSquaredLogarithmicError(),
            ]

        self.network.compile(
            optimizer=optimizer(learning_rate=rate),
            loss=loss_func,
            metrics=metrics,
            run_eagerly=False,
        )

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

        return self.network(inputs)

    def train(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        validation_split=0.0,
        epochs=50,
        mini_batch_size=None,
        callbacks=None,
        verbose="auto",
    ) -> keras.callbacks.History:
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
        self._train_history = self.network.fit(
            x_data,
            y_data,
            batch_size=mini_batch_size,
            callbacks=callbacks,
            validation_split=validation_split,
            epochs=epochs,
            verbose=verbose,
        )
        return self._train_history

    def export_to_cpp(
        self, path: str, array_type: str = "[]", path_to_compiler: str = "g++", **kwargs
    ):
        res = """
        #include <cmath>
        #include <vector>

        """

        config = self.to_dict(**kwargs)

        input_size = self._input_size
        output_size = self._output_size
        blocks = self._shape
        layers = config["layer"]

        comment = f"// This function takes {input_size} elements array and returns {output_size} elements array\n"
        signature = f""
        start_func = "{\n"
        end_func = "}\n"
        transform_input_vector = ""
        transform_output_array = ""
        return_stat = "return answer;\n"

        creator_1d: Callable[[str, int], str] = cpp_utils.array1d_creator("float")
        creator_heap_1d: Callable[[str, int], str] = cpp_utils.array1d_heap_creator(
            "float"
        )
        creator_2d: Callable[[str, int, int], str] = cpp_utils.array2d_creator("float")
        if array_type == "[]":
            signature = f"float* feedforward(float x_array[])\n"

        if array_type == "vector":
            signature = f"vector<float> feedforward(vector<float> x)\n"

            transform_input_vector = cpp_utils.transform_1dvector_to_array(
                "float", input_size, "x", "x_array"
            )
            transform_output_array = cpp_utils.transform_1darray_to_vector(
                "float", output_size, "", "answer_vector"
            )
            return_stat = "return answer_vector;\n"

        create_layers = ""
        create_layers += creator_1d(f"layer_0", input_size, initial_value=0)
        for i, size in enumerate(blocks):
            create_layers += creator_1d(f"layer_{i + 1}", size, initial_value=0)
        create_layers += creator_1d(
            f"layer_{len(blocks) + 1}", output_size, initial_value=0
        )
        create_layers += cpp_utils.copy_1darray_to_array(
            input_size, "x_array", "layer_0"
        )

        create_weights = ""
        for i, layer_dict in enumerate(layers):
            create_weights += creator_2d(
                f"weight_{i}_{i + 1}",
                layer_dict[LAYER_DICT_NAMES["inp_size"]],
                layer_dict[LAYER_DICT_NAMES["shape"]],
            )

        fill_weights = ""
        for i, layer_dict in enumerate(layers):
            fill_weights += cpp_utils.fill_2d_array_by_list(
                f"weight_{i}_{i + 1}", layer_dict[LAYER_DICT_NAMES["weights"]]
            )

        create_biases = ""
        for i, layer_dict in enumerate(layers):
            create_biases += creator_1d(
                f"bias_{i + 1}", layer_dict[LAYER_DICT_NAMES["shape"]]
            )

        fill_biases = ""
        for i, layer_dict in enumerate(layers):
            fill_biases += cpp_utils.fill_1d_array_by_list(
                f"bias_{i + 1}", layer_dict[LAYER_DICT_NAMES["bias"]]
            )
            # fill_biases += cpp_utils.fill_1d_array_by_list_short("float",
            #                                                      layer_dict[LAYER_DICT_NAMES["shape"]],
            #                                                      f"bias_{i + 1}",
            #                                                      f"temp_{i + 1}",
            #                                                      layer_dict[LAYER_DICT_NAMES["bias"]])

        feed_forward_cycles = ""
        for i, layer_dict in enumerate(layers):
            left_size = layer_dict[
                LAYER_DICT_NAMES["inp_size"]
            ]  # if i != 0 else input_size
            right_size = layer_dict[LAYER_DICT_NAMES["shape"]]
            act_func = layer_dict[LAYER_DICT_NAMES["activation"]]
            decorator_params = layer_dict.get(LAYER_DICT_NAMES["decorator_params"])
            feed_forward_cycles += cpp_utils.feed_forward_step(
                f"layer_{i}",
                left_size,
                f"layer_{i + 1}",
                right_size,
                f"weight_{i}_{i + 1}",
                f"bias_{i + 1}",
                act_func,
                decorator_params,
            )

        move_result = creator_heap_1d("answer", output_size)
        move_result += cpp_utils.copy_1darray_to_array(
            output_size, f"layer_{len(blocks) + 1}", "answer"
        )

        res += comment
        res += signature
        res += start_func
        res += transform_input_vector
        res += create_layers
        res += create_weights
        res += fill_weights
        res += create_biases
        res += fill_biases
        res += feed_forward_cycles
        res += move_result
        res += transform_output_array
        res += return_stat
        res += end_func

        header_res = f"""
#ifndef {path[0].upper() + path[1:]}_hpp
#define {path[0].upper() + path[1:]}_hpp

{comment}
{signature}

#endif /* {path[0].upper() + path[1:]}_hpp */

        """

        with open(path + ".cpp", "w") as f:
            f.write(res)

        with open(path + ".hpp", "w") as f:
            f.write(header_res)

        os.system(path_to_compiler + " -c -Ofast " + path + ".cpp")

    def to_dict(self, **kwargs):
        return self.network.to_dict(**kwargs)

    def export_to_file(self, path, **kwargs):
        config = self.to_dict(**kwargs)
        with open(path + ".apg", "w") as f:
            f.write(HEADER_OF_FILE + json.dumps(config))

    def from_file(self, path, **kwargs):
        with open(path + ".apg", "r") as f:
            for header in range(HEADER_OF_FILE.count("\n")):
                # TODO: check version compability
                _ = f.readline()
            config = json.loads(f.readline())
            self.network.from_dict(config)
            self.set_name(config["name"])

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
    def create_neuron(cls, input_size, output_size, shape, **kwargs):
        activation, decorator_params, weight, biases, kwargs = _get_act_and_init(
            kwargs,
            keras.activations.sigmoid,
            None,
            tf.random_normal_initializer(),
        )

        res = cls(
            input_size=input_size,
            block_size=shape,
            output_size=output_size,
            activation_func=activation,
            bias_init=biases,
            weight_init=weight,
            decorator_params=decorator_params,
            **kwargs,
        )

        return res

    @classmethod
    def create_perceptron(cls, input_size, output_size, shape, threshold=1, **kwargs):
        activation, decorator_params, weight, biases, kwargs = _get_act_and_init(
            kwargs,
            activations.perceptron_threshold,
            [{"threshold": threshold}],
            tf.random_normal_initializer(),
        )

        res = cls(
            input_size=input_size,
            block_size=shape,
            output_size=output_size,
            activation_func=activation,
            bias_init=biases,
            weight_init=weight,
            decorator_params=decorator_params,
            **kwargs,
        )

        return res


_create_functions = defaultdict(lambda: DenseNet)
_create_functions["DenseNet"] = DenseNet
