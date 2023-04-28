"""
Module for training neural networks
"""
import gc
import random
import time
from typing import Union, Tuple, List, Dict, Any

import numpy as np
import tensorflow as tf
import keras.backend as k
from memory_profiler import profile

from networks import imodel, activations

from networks import losses
from networks.callbacks import MemoryCleaner, MeasureTrainTime, LightHistory
from networks.losses import get_loss
from networks.metrics import get_metric

_default_shapes = [
    [10, 10, 10, 10, 10, 10],
    [80, 80, 80],
    [32, 16, 8, 4],
    [4, 8, 16, 32],
    [10, 10],
    [30],
]


def _create_random_network(
    inp: int,
    out: int,
    min_layers=1,
    max_layers=5,
    min_neurons=3,
    max_neurons=60,
    args=None,
) -> list[Union[list[int], list, list[Union[str, Any]], list[None]]]:
    """
    Create random neural network from the passed parameters.

    Parameters
    ----------
    inp: int
        Amount of input variables
    out: int
        Amount of output variables
    min_layers: int
        Minimal count of layers in neural net
    max_layers: int
        Maximal count of layers in neural net
    min_neurons: int
        Minimal count of neurons per layer
    max_neurons: int
        Maximal count of neurons per layer
    Returns
    -------
    net: network.INetwork
        Random neural network
    """

    layers = random.randint(min_layers, max_layers)
    shape = [random.randint(min_neurons, max_neurons) for _ in range(layers)]
    act = []
    act_names = []
    decorator_param = []
    all_act_names = list(activations.get_all_activations().keys())
    for _ in shape:
        act_names.append(random.choice(all_act_names))
        act.append(activations.get(act_names[-1]))
        # TODO: activation func can take additional arguments
        # but at this moment I dont create random arguments (instead of *None* in decorator_params)
        decorator_param.append(None)
    act_names.append("linear")
    act.append(activations.get("linear"))
    decorator_param.append(None)

    nets_param = [shape, act, act_names, decorator_param]
    return nets_param


def _normalize_two_array(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float]:
    m = max(abs(np.amax(x)), abs(np.amax(y)))
    if m < 1:
        m = 1
    x = x / m
    y = y / m
    return x, y, m


def _normalize_array(x: np.ndarray) -> Tuple[np.ndarray, float]:
    m = abs(np.amax(x))
    if m != 0:
        x = x / m
    return x, m


# @profile
def train(
    x_data: np.ndarray, y_data: np.ndarray, **kwargs
) -> tuple[imodel.IModel, dict[Union[str, Any], Any]]:
    """
    Choose and return neural network which present the minimal average absolute deviation.
    x_data and y_data is numpy 2d arrays (in case we don't have multiple-layer input/output).

    Parameters
    ----------
    x_data: np.ndarray
        Array of inputs --- [input1, input2, ...]
    y_data: np.ndarray
        List of outputs --- [output1, output2, ...]
    Returns
    -------
    net: imodel.IModel
        Best neural network for this dataset
    history: Dict[str, list]
        History of training for this network
    """
    # default config
    args = {
        "debug": False,
        "eps": 1e-2,
        "epochs": 100,
        "validation_split": 0.2,
        "normalize": False,
        "name_salt": "",
        "loss_func": get_loss("MeanSquaredError"),
        "optimizer": tf.keras.optimizers.SGD,
        "metrics": [
            get_metric("MeanSquaredError"),
            get_metric("MeanAbsoluteError"),
            get_metric("CosineSimilarity"),
        ],
        "validation_metrics": [
            get_metric("MeanSquaredError"),
            get_metric("MeanAbsoluteError"),
            get_metric("CosineSimilarity"),
        ],
        "use_rand_net": True,
        "experiments": False,
        "net_type": "DenseNet",
        "nets_param": [
            [
                shape,  # shape
                [activations.get("sigmoid")] * len(shape)
                + [activations.get("linear")],  # activation functions
                ["sigmoid"] * len(shape) + ["linear"],  # activation names
                [None] * (len(shape) + 1),  # decorator parameters for activation
            ]
            for shape in _default_shapes
        ],
    }
    for kw in kwargs:
        args[kw] = kwargs[kw]

    if args["debug"]:
        print("Start train func")

    # determining the number of inputs and outputs of the neural network
    if type(x_data[0]) is np.ndarray:
        input_len = len(x_data[0])
    else:
        input_len = 1

    if type(y_data[0]) is np.ndarray:
        output_len = len(y_data[0])
    else:
        output_len = 1

    # prepare data (normalize)
    norm_coff = 1
    if args["normalize"]:
        x_data, norm_coff = _normalize_array(x_data)
        # y_data, norm_coff = _normalize_array(y_data)
        if args["debug"]:
            print(f"Normalization coefficient is {norm_coff}")

    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)

    # prepare neural networks
    if args["debug"]:
        print("Prepare neural networks and data")
    nets = []
    for parameters in args["nets_param"]:
        shape = parameters[0]
        act = parameters[1]
        act_names = parameters[2]
        decorator_param = parameters[3]
        str_shape = "_".join(map(str, shape))
        curr_net = imodel.IModel(
            input_size=input_len,
            block_size=shape,
            output_size=output_len,
            activation_func=act,
            activation_names=act_names,
            decorator_params=decorator_param,
            net_type=args["net_type"],
            name=f"net{args['name_salt']}_{str_shape}",
            is_debug=args["debug"],
        )
        nets.append(curr_net)
    if args["use_rand_net"]:
        rand_net = _create_random_network(input_len, output_len, args=args)
        nets.append(rand_net)

    # compile
    for nn in nets:
        nn.compile(
            rate=args["eps"],
            optimizer=args["optimizer"],
            loss_func=args["loss_func"],
            metrics=args["metrics"],
            # run_eagerly=True,
        )

    if args["debug"]:
        print("Success prepared")

    # train
    history = []
    for i, nn in enumerate(nets):
        verb = 0
        if args["debug"]:
            print(nn)
            verb = 1
        temp_his = nn.train(
            x_data,
            y_data,
            epochs=args["epochs"],
            validation_split=args["validation_split"],
            callbacks=[MemoryCleaner(), MeasureTrainTime()],
            verbose=verb,
        )
        temp_last_res = dict()
        for key in temp_his.history:
            temp_last_res[key] = temp_his.history[key].copy()
        temp_last_res["train_time"] = nn.network.trained_time["train_time"]
        temp_last_res["epoch_time"] = nn.network.trained_time["epoch_time"].copy()

        history.append(temp_last_res)
    result_net = nets[0]
    result_history = history[0]
    min_err = history[0]["loss"]
    for i in range(1, len(nets)):
        if history[i]["loss"] < min_err:
            min_err = history[i]["loss"]
            result_net = nets[i]
            result_history = history[i]
    if args["debug"]:
        print(f"Minimal loss error is {min_err} {args['name_salt']}")
    return result_net, result_history


# @profile
def experiments_train(
    x_data: np.ndarray, y_data: np.ndarray, val_data=None, **kwargs
) -> tuple[dict[str, list], list[dict[str, float]], list[dict[str, float]]]:
    """
    Choose and return neural network which present the minimal average absolute deviation.
    x_data and y_data is numpy 2d arrays (in case we don't have multiple-layer input/output).

    Parameters
    ----------
    x_data: np.ndarray
        Array of inputs --- [input1, input2, ...]
    y_data: np.ndarray
        List of outputs --- [output1, output2, ...]
    val_data: tuple[np.ndarray, np.ndarray]
        Validation data
    Returns
    -------
    net: network.INetwork
        Best neural network for this dataset
    history: tf.keras.callbacks.History
        History of training for this network
    """
    # default config
    args = {
        "debug": False,
        "eps": 1e-2,
        "epochs": 100,
        "validation_split": 0.2,
        "normalize": False,
        "name_salt": "",
        "loss_func": get_loss("MeanSquaredError"),
        "optimizer": tf.keras.optimizers.SGD,
        "metrics": [
            get_metric("MeanSquaredError"),
            get_metric("MeanAbsoluteError"),
            get_metric("CosineSimilarity"),
        ],
        "validation_metrics": [
            get_metric("MeanSquaredError"),
            get_metric("MeanAbsoluteError"),
            get_metric("CosineSimilarity"),
        ],
        "use_rand_net": True,
        "experiments": True,
        "net_type": "DenseNet",
        "nets_param": [
            [
                shape,  # shape
                [activations.get("sigmoid")] * len(shape)
                + [activations.get("linear")],  # activation functions
                ["sigmoid"] * len(shape) + ["linear"],  # activation names
                [None] * (len(shape) + 1),  # decorator parameters for activation
            ]
            for shape in _default_shapes
        ],
    }
    for kw in kwargs:
        args[kw] = kwargs[kw]

    # determining the number of inputs and outputs of the neural network
    if type(x_data[0]) is np.ndarray:
        input_len = len(x_data[0])
    else:
        input_len = 1

    if type(y_data[0]) is np.ndarray:
        output_len = len(y_data[0])
    else:
        output_len = 1

    # prepare data (normalize)
    norm_coff = 1
    if args["normalize"]:
        x_data, norm_coff = _normalize_array(x_data)

    x_data = tf.convert_to_tensor(x_data, dtype=tf.float32)
    y_data = tf.convert_to_tensor(y_data, dtype=tf.float32)

    nets = []
    for parameters in args["nets_param"]:
        shape = parameters[0]
        act = parameters[1]
        act_names = parameters[2]
        decorator_param = parameters[3]
        str_shape = "_".join(map(str, shape))
        curr_net = imodel.IModel(
            input_size=input_len,
            block_size=shape,
            output_size=output_len,
            activation_func=act,
            activation_names=act_names,
            decorator_params=decorator_param,
            net_type=args["net_type"],
            name=f"net{args['name_salt']}_{str_shape}",
        )
        nets.append(curr_net)
    if args["use_rand_net"]:
        rand_net_params = _create_random_network(input_len, output_len, args=args)
        rand_net = imodel.IModel(
            input_size=input_len,
            block_size=rand_net_params[0],
            output_size=output_len,
            activation_func=rand_net_params[1],
            activation_names=rand_net_params[2],
            decorator_params=rand_net_params[3],
            net_type=args["net_type"],
            name=f"net{args['name_salt']}_random",
        )
        nets.append(rand_net)

    # compile
    for nn in nets:
        nn.compile(
            rate=args["eps"],
            optimizer=args["optimizer"],
            loss_func=args["loss_func"],
            metrics=args["metrics"],
            # run_eagerly=True,
        )

    # train
    history = []
    val_history = []
    nets_param = {"shape": [], "activations": []}
    for _, nn in enumerate(nets):
        time_viewer = MeasureTrainTime()
        verb = 0
        if args["debug"]:
            print(nn)
            verb = 1
        temp_his = nn.train(
            x_data,
            y_data,
            epochs=args["epochs"],
            validation_split=args["validation_split"],
            callbacks=[LightHistory(), time_viewer],
            verbose=verb,
        )
        temp_his_last_res = dict()
        for key in temp_his.history:
            temp_his_last_res[key] = temp_his.history[key]
        temp_his_last_res["train_time"] = nn.network.trained_time["train_time"]

        history.append(temp_his_last_res)

        if val_data is not None:
            val_max_size = 40_000
            predict_max_size = 100
            size = val_data[0].shape

            val_x = np.concatenate([val_data[0]] * (val_max_size // size[0]))
            val_y = np.concatenate([val_data[1]] * (val_max_size // size[0]))

            predict_x = np.split(val_data[0][0:predict_max_size], predict_max_size)
            predict_y = np.split(val_data[1][0:predict_max_size], predict_max_size)

            validation_history = nn.evaluate(
                val_x,
                val_y,
                callbacks=[MemoryCleaner(), time_viewer],
                verbose=verb,
                return_dict=True,
            )
            temp_val_last_res = dict()
            for key in validation_history:
                temp_val_last_res["predict_" + key] = validation_history[key]
            temp_val_last_res["predict_all_time"] = nn.network.trained_time[
                "predict_time"
            ]

            temp_val_last_res["predict_single_time"] = 0

            for row in predict_x:
                nn.network.predict(x=row, verbose=verb, callbacks=[time_viewer])
                temp_val_last_res["predict_single_time"] += nn.network.trained_time[
                    "predict_time"
                ]
            val_history.append(temp_val_last_res)

        nets_param["shape"].append(
            str([nn.get_input_size] + nn.get_shape + [nn.get_output_size])
        )
        acts = nn.get_activations
        acts_name = []
        for i in range(len(acts)):
            acts_name.append(acts[i].__name__)
        nets_param["activations"].append(str(acts_name))

        gc.collect()
        k.clear_session()

    return nets_param, history, val_history


# @profile
def full_search(
    x_data: np.ndarray, y_data: np.ndarray, x_val=None, y_val=None, **kwargs
) -> list[list[dict, float, float, imodel.IModel]]:
    """
    Choose and return neural network which present the minimal average absolute deviation.
    x_data and y_data is numpy 2d arrays (in case we don't have multiple-layer input/output).

    Parameters
    ----------
    x_data: np.ndarray
        Array of inputs --- [input1, input2, ...]
    y_data: np.ndarray
        List of outputs --- [output1, output2, ...]
    Returns
    -------
    net: network.INetwork
        Best neural network for this dataset
    """

    # default config
    args = {
        "rates": [1e-2, 5e-3, 1e-3],
        # "epochs_data": [50, 200],
        # "split_sizes": [0.2],
        # "normalize_data": [False, True],
        "losses_functions": losses.get_all_loss_functions(),
        "optimizers": [
            tf.keras.optimizers.SGD,
            tf.keras.optimizers.Adam,
            tf.keras.optimizers.RMSprop,
        ],
        "metrics": [
            losses.get_all_metric_functions()[key]
            for key in losses.get_all_metric_functions()
        ],
        "validation_metrics": [
            losses.get_all_metric_functions()[key]
            for key in losses.get_all_metric_functions()
        ],
        "nets_shape": _default_shapes,
        # "rates": [1e-2],
        "epochs_data": [50],
        "validation_split": [0.2],
        "normalize_data": [False],
        # "losses_functions": {"mse": tf.keras.losses.MeanSquaredError()},
        # "optimizers": [tf.keras.optimizers.SGD],
    }

    for arg in args:
        if kwargs.get(arg) is not None:
            args[arg] = kwargs[arg]
            kwargs.pop(arg)

    val_data = None
    if x_val is not None and y_val is not None:
        val_data = (x_val, y_val)

    # Networks parameters --- shape and activation functions

    activation_funcs = activations.get_all_activations()
    nets_param = []
    for shape in args["nets_shape"]:
        if len(shape) != 0:
            for activation in activation_funcs:
                nets_param.append(
                    [
                        shape,
                        [activations.get(activation)] * len(shape)
                        + [activations.get("linear")],
                        [activation] * len(shape) + ["linear"],
                        [None] * (len(shape) + 1),
                    ]
                )
        else:
            nets_param.append(
                [
                    shape,
                    [activations.get("linear")],
                    ["linear"],
                    [None],
                ]
            )

    metaparams = []
    for loss_func in args["losses_functions"]:
        for normalize in args["normalize_data"]:
            for optimizer in args["optimizers"]:
                for validation_split in args["validation_split"]:
                    for epochs in args["epochs_data"]:
                        for rate in args["rates"]:
                            metaparams.append(dict())
                            metaparams[-1]["loss_func"] = args["losses_functions"][
                                loss_func
                            ]
                            metaparams[-1]["normalize"] = normalize
                            metaparams[-1]["optimizer"] = optimizer
                            metaparams[-1]["validation_split"] = validation_split
                            metaparams[-1]["epochs"] = epochs
                            metaparams[-1]["eps"] = rate
                            metaparams[-1]["metrics"] = args["metrics"]
                            metaparams[-1]["validation_metrics"] = args[
                                "validation_metrics"
                            ]
                            metaparams[-1]["nets_param"] = nets_param
                            metaparams[-1].update(kwargs)

    best_nets: List[List[dict, float, float, imodel.IModel]] = []
    print(len(metaparams))
    print(len(nets_param))
    for i, params in enumerate(metaparams):
        print(i)
        if kwargs.get("experiments"):
            temp_train_results = experiments_train(
                x_data, y_data, val_data=val_data, **params
            )
            trained: dict[str, list] = temp_train_results[0]
            history = temp_train_results[1]
            val_history = temp_train_results[2]
            best_nets.append([params, trained, history, val_history])
        else:
            trained, history = train(x_data, y_data, val_data=val_data, **params)
            loss = history["loss"][-1]
            val_loss = history["val_loss"][-1]
            best_nets.append([params, loss, val_loss, trained])

    if kwargs.get("experiments"):
        return best_nets

    best_nets.sort(key=lambda x: [x[1], x[2]])
    return best_nets


def _load_network_shapes():
    return [
        [10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [4, 8, 16, 32],
        [32, 16, 8, 4],
        [100, 100, 100],
        [],
    ]
    # res = []
    # with open("../resource/network_shapes.txt", "r") as f:
    #     for line in f.readlines():
    #         res.append(list(map(int, line.split())))
    # return res


temp = _load_network_shapes()
if len(temp) > 0:
    _default_shapes = temp
