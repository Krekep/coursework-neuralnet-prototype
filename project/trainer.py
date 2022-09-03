"""
Module for training neural networks
"""
import random
import numpy as np

from project.networks import inetwork

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
                            reg_abs=0,
                            reg_quad=0
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
    reg_abs: int
        L1 regularization
    reg_quad: int
        L2 regularization
    Returns
    -------
    net: network.INetwork
        Random neural network
    """

    layers = random.randint(min_layers, max_layers)
    shape = [random.randint(min_neurons, max_neurons) for _ in range(layers)]
    net = inetwork.INetwork(input_size=inp, block_size=shape, output_size=out)
    return net


def train(x_data: np.ndarray, y_data: np.ndarray, debug=False) -> inetwork.INetwork:
    """
    Choose and return neural network which present the minimal average absolute deviation.
    x_data and y_data is numpy 2d arrays (in case we don't have multiple-layer input/output).

    Parameters
    ----------
    x_data: np.ndarray
        Array of inputs --- [input1, input2, ...]
    y_data: np.ndarray
        List of outputs --- [output1, output2, ...]
    debug: bool
        Is debug output enabled
    Returns
    -------
    net: network.INetwork
        Best neural network for this dataset
    """
    if debug:
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

    # prepare neural networks
    if debug:
        print("Prepare neural networks and data")
    nets = []
    for shape in _default_shapes:
        curr_net = inetwork.INetwork(input_size=input_len, block_size=shape, output_size=output_len)
        nets.append(curr_net)
    rand_net = _create_random_network(input_len, output_len)
    nets.append(rand_net)
    if debug:
        print("Success prepared")

    # train
    history = [0] * len(nets)
    for i, nn in enumerate(nets):
        verb = 0
        if debug:
            print(nn)
            verb = 1
        history[i] = nn.train(x_data, y_data, epochs=100, validation_split=0.2, verbose=verb)
    print("Success train")
    result_net = nets[0]
    min_err = history[0].history["loss"][-1]
    for i in range(1, len(nets)):
        if history[i].history["loss"][-1] < min_err:
            min_err = history[i].history["loss"][-1]
            result_net = nets[i]
    print("Minimal quad average error is", min_err)
    return result_net
