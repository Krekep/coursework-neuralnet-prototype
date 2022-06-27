"""
Module for training neural networks
"""
import random
import numpy as np
from typing import List, Tuple

import project
from project import network

_default_shapes = [
    [10, 10, 10, 10, 10, 10],
    [200, 200, 200],
    [32, 16, 8, 4],
    [4, 8, 16, 32],
    [10, 10],
    [30]
]

# additional variables, must be deleted?
_scalar_1 = np.ndarray((1,))
_scalar_2 = np.ndarray((1, 1))


def _prepare_cross_validation(x_data: List, y_data: List, parts_num: int) -> Tuple[List, List]:
    """
    Returns data combined in num_parts - 1 chunks for training and
     a chunk of data for testing (no training is performed on it)

    Parameters
    ----------
    x_data: List
        List of inputs
    y_data: List
        List of outputs
    parts_num: int
        Count of chunks
    Returns
    -------
    splitting_data: Tuple[List, List]
        Data for training and data for testing the result of training
    """

    result_data = [0] * (parts_num - 1)
    size_of_parts = len(x_data) // parts_num
    for i in range(parts_num - 1):
        result_data[i] = [x_data[(i * size_of_parts):((i + 1) * size_of_parts)],
                          y_data[(i * size_of_parts):((i + 1) * size_of_parts)]]
    test_data = [x_data[((parts_num - 1) * size_of_parts):(parts_num * size_of_parts)],
                 y_data[((parts_num - 1) * size_of_parts):(parts_num * size_of_parts)]]
    return result_data, test_data


def _create_random_network(inp: int,
                           out: int,
                           min_layers=1,
                           max_layers=5,
                           min_neurons=3,
                           max_neurons=60
                           ) -> network.Network:
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
    net: network.Network
        Random neural network
    """

    layers = random.randint(min_layers, max_layers)
    shape = [inp] + [random.randint(min_neurons, max_neurons) for _ in range(layers)] + [out]
    net = network.Network(shape)
    return net


def train(x_data: List, y_data: List) -> network.Network:
    """
    Choose and return neural network which present the minimal average absolute deviation

    Parameters
    ----------
    x_data: List
        List of inputs
    y_data: List
        List of outputs
    Returns
    -------
    net: network.Network
        Best neural network for this dataset
    """
    # determining the number of inputs and outputs of the neural network
    if type(x_data[0]) is np.ndarray:
        input_len = len(x_data[0])
    else:
        input_len = 1

    if type(y_data[0]) is np.ndarray:
        output_len = len(y_data[0])
    else:
        output_len = 1

    # prepare neural networks and data
    data, test_data = _prepare_cross_validation(x_data, y_data, 5)
    nets = []
    for i in range(len(_default_shapes)):
        temp_shape = [input_len] + _default_shapes[i] + [output_len]
        nets.append(network.Network(temp_shape))
    nets.append(_create_random_network(input_len, output_len))

    # train
    avg_errors = [0] * len(nets)
    for validation_piece in range(len(data)):
        train_data_x = np.array([])
        train_data_y = np.array([])
        for test_piece in range(len(data)):
            if test_piece != validation_piece:
                train_data_x = np.vstack([train_data_x, data[test_piece][0]]) if train_data_x.size else \
                data[test_piece][0]
                train_data_y = np.vstack([train_data_y, data[test_piece][1]]) if train_data_y.size else \
                data[test_piece][1]
        train_data = np.hstack([train_data_x, train_data_y])

        for nn in nets:
            nn.SGD(training_data=train_data, epochs=50,
                   mini_batch_size=max(len(train_data) // 10, 1), eta=0.02)
        validation_data = np.hstack([data[validation_piece][0], data[validation_piece][1]])
        for nn in nets:
            nn.SGD(training_data=validation_data, epochs=15,
                   mini_batch_size=max((len(data[validation_piece]) // 10), 1), eta=0.05)

        # count the errors
        temp = [0] * len(nets)
        for i in range(len(nets)):
            for example in range(len(test_data[0])):
                if test_data[0][example].shape == _scalar_1.shape:
                    predicted = nets[i].feedforward(float(test_data[0][example]))
                else:
                    predicted = nets[i].feedforward(test_data[0][example])
                if predicted.shape == _scalar_2.shape:
                    temp[i] += abs(float(predicted) - float(test_data[1][example]))
                else:
                    temp[i] += abs(sum(predicted) - sum(test_data[1][example]))
            avg_errors[i] += temp[i] / len(test_data[0])

    # find the best net
    result_net = nets[0]
    min_err = avg_errors[0]
    for i in range(1, len(nets)):
        if avg_errors[i] < min_err:
            min_err = avg_errors[i]
            result_net = nets[i]
    print("Minimal average error is", min_err)
    return result_net
