"""
Module for training neural networks
"""
import random
from typing import Union

import numpy as np
import tensorflow as tf

from project.networks import inetwork

from project.networks import losses
from project.networks.losses import MyMSE

_default_shapes = [
    [10, 10, 10, 10, 10, 10],
    [80, 80, 80],
    [32, 16, 8, 4],
    [4, 8, 16, 32],
    [10, 10],
    [30]
]


def _create_random_network(inp: int,
                           out: int,
                           min_layers=1,
                           max_layers=5,
                           min_neurons=3,
                           max_neurons=60,
                           args=None
                           ) -> inetwork.INetwork:
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
    str_shape = "_".join(map(str, shape))
    net = inetwork.INetwork(input_size=inp, block_size=shape, output_size=out, rate=args["eps"],
                            optimizer=args["optimizer"], loss_func=args["loss_func"], metrics=args["metrics"],
                            name=f"net{args['name_salt']}_{str_shape}", is_debug=args["debug"])
    return net


def _normalize_two_array(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    m = max(np.amax(x), np.amax(y))
    x = x / m
    y = y / m
    return x, y, m


def _normalize_array(x: np.ndarray) -> tuple[np.ndarray, float]:
    m = np.amax(x)
    x = x / m
    return x, m


def train(x_data: np.ndarray, y_data: np.ndarray, **kwargs) -> Union[
    inetwork.INetwork, tuple[list[inetwork.INetwork], tf.keras.callbacks.History]
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
    """
    # default config
    args = {
        "debug": False,
        "eps": 1e-2,
        "epochs": 100,
        "validation_split": 0.2,
        "normalize": False,
        "name_salt": "",
        "loss_func": MyMSE(),
        "optimizer": tf.keras.optimizers.SGD,
        "metrics": [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(),
                    tf.keras.metrics.CosineSimilarity()],
        "use_rand_net": True,
        "experiments": False
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
        y_data, norm_coff = _normalize_array(x_data)

    # prepare neural networks
    if args["debug"]:
        print("Prepare neural networks and data")
    nets = []
    for shape in _default_shapes:
        str_shape = "_".join(map(str, shape))
        curr_net = inetwork.INetwork(input_size=input_len, block_size=shape, output_size=output_len, rate=args["eps"],
                                     optimizer=args["optimizer"], loss_func=args["loss_func"], metrics=args["metrics"],
                                     normalization=norm_coff, name=f"net{args['name_salt']}_{str_shape}",
                                     is_debug=args["debug"])
        nets.append(curr_net)
    if args["use_rand_net"]:
        rand_net = _create_random_network(input_len, output_len, args=args)
        nets.append(rand_net)
    if args["debug"]:
        print("Success prepared")

    # train
    history = [0] * len(nets)
    for i, nn in enumerate(nets):
        verb = 0
        if args["debug"]:
            print(nn)
            verb = 1
        history[i] = nn.train(x_data, y_data, epochs=args["epochs"], validation_split=args["validation_split"],
                              callbacks=tf.keras.callbacks.CSVLogger(f"log_{nn.get_name}.csv",
                                                                     separator=',', append=False), verbose=verb)
    result_net = nets[0]
    min_err = history[0].history["loss"][-1]
    for i in range(1, len(nets)):
        if history[i].history["loss"][-1] < min_err:
            min_err = history[i].history["loss"][-1]
            result_net = nets[i]
    if args["debug"]:
        print(f"Minimal quad average error is {min_err} {args['name_salt']}")
    if args["experiments"]:
        return nets, history
    return result_net


def _load_network_shapes():
    res = []
    with open("../resource/network_shapes.txt", "r") as f:
        for line in f.readlines():
            res.append(list(map(int, line.split())))
    return res


temp = _load_network_shapes()
if len(temp) > 0:
    _default_shapes = temp
