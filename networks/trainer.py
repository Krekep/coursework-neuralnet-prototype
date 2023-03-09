"""
Module for training neural networks
"""
import random
from typing import Union, Tuple, List

import numpy as np
import tensorflow as tf

from networks import imodel, activations

from networks import losses
from networks.losses import get_loss, get_metric

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
) -> imodel.IModel:
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
        # TODO: activation func can take additional arguments
        # but at this moment I dont create random arguments (insted of *None* in decorator_params)
        decorator_param.append(None)
    act_names.append("linear")
    act.append(activations.get("linear"))
    decorator_param.append(None)
    str_shape = "_".join(map(str, shape))
    net = imodel.IModel(
        input_size=inp,
        block_size=shape,
        output_size=out,
        activation_func=act,
        activation_names=act_names,
        decorator_params=decorator_param,
        net_type=args["net_type"],
        name=f"net{args['name_salt']}_{str_shape}",
        is_debug=args["debug"],
    )
    return net


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
    if m < 1:
        m = 1
    x = x / m
    return x, m


def train(
    x_data: np.ndarray, y_data: np.ndarray, **kwargs
) -> Union[
    Tuple[imodel.IModel, tf.keras.callbacks.History],
    Tuple[List[imodel.IModel], tf.keras.callbacks.History],
]:
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
        "activation_func": activations.get("linear"),
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
        history[i].append(
            nn.train(
                x_data,
                y_data,
                epochs=args["epochs"],
                validation_split=args["validation_split"],
                callbacks=tf.keras.callbacks.CSVLogger(
                    f"log_{nn.get_name}.csv", separator=",", append=False
                ),
                verbose=verb,
            )
        )
    result_net = nets[0]
    result_history = history[0]
    min_err = history[0].history["loss"][-1]
    for i in range(1, len(nets)):
        if history[i].history["loss"][-1] < min_err:
            min_err = history[i].history["loss"][-1]
            result_net = nets[i]
            result_history = result_history[i]
    if args["debug"]:
        print(f"Minimal loss error is {min_err} {args['name_salt']}")
    if args["experiments"]:
        return nets, history
    return result_net, result_history


def full_search(
    x_data: np.ndarray, y_data: np.ndarray, **kwargs
) -> list[list[float, float, imodel.IModel]]:
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

    # Networks parameters --- shape and activation functions
    nets_shape = _default_shapes
    activation_funcs = activations.get_all_activations()
    nets_param = [
        [
            shape,
            [activations.get(activation)] * len(shape) + [activations.get("linear")],
            [activation] * len(shape) + ["linear"],
            [None] * (len(shape) + 1),
        ]
        for activation in activation_funcs
        for shape in nets_shape
    ]

    # Training algorithms
    # optimizers = [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop]
    optimizers = [tf.keras.optimizers.SGD]

    # Should we do data normalize before training
    # normalize_data = [False, True]
    normalize_data = [False]

    # Full train iteration over data
    epochs_data = [50]
    # epochs_data = [50, 200, 500, 1000, 2000]

    # Learning step
    # rates = [1e-2, 5e-3, 1e-3]
    rates = [1e-2]

    # Validation metrics
    validation_metrics = [
        losses.get_all_metric_functions()[key]
        for key in losses.get_all_metric_functions()
    ]

    # Loss function for training
    # losses_functions = losses.get_all_loss_functions()
    losses_functions = {"MeanSquaredError": losses.get_loss("MeanSquaredError")}

    # Training metrics
    metrics = [
        losses.get_all_metric_functions()[key]
        for key in losses.get_all_metric_functions()
    ]

    # How much percentage of samples from data will be used for validation
    split_sizes = [0.2]

    metaparams = []
    for loss_func in losses_functions:
        for normalize in normalize_data:
            for optimizer in optimizers:
                for validation_split in split_sizes:
                    for epochs in epochs_data:
                        for rate in rates:
                            metaparams.append(dict())
                            metaparams[-1]["loss_func"] = losses_functions[loss_func]
                            metaparams[-1]["normalize"] = normalize
                            metaparams[-1]["optimizer"] = optimizer
                            metaparams[-1]["validation_split"] = validation_split
                            metaparams[-1]["epochs"] = epochs
                            metaparams[-1]["rate"] = rate
                            metaparams[-1]["metrics"] = metrics
                            metaparams[-1]["validation_metrics"] = validation_metrics
                            metaparams[-1]["nets_param"] = nets_param
                            metaparams[-1].update(kwargs)

    best_nets: List[List[float, float, imodel.IModel]] = []
    for i, params in enumerate(metaparams):
        trained, history = train(x_data, y_data, **params)
        loss = history.history["loss"][-1]
        val_loss = history.history["val_loss"][-1]
        best_nets.append([loss, val_loss, trained])

    best_nets.sort(key=lambda x: [x[0], x[1]])

    return best_nets


def _load_network_shapes():
    return [
        [10, 10, 10],
        [10, 10, 10, 10, 10, 10],
        [4, 8, 16, 32, 64],
        [4, 8, 16],
        [64, 32, 16, 8, 4],
        [16, 8, 4],
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
