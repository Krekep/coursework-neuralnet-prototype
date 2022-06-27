"""
Provide some helpful functions
"""

import random
from typing import Tuple

import project
from project import network

import pickle
import numpy as np
import matplotlib.pyplot as plt


def export_network(path: str, net: network.Network) -> None:
    """
    This method saves the neural network to a file
    using the pickle library functions.

    Parameters
    ----------
    path: str
        Path to file
    net: network.Network
        Neural network to be saved
    Returns
    -------
    None
    """
    with open(path, "wb") as file:
        pickle.dump(net, file)


def import_network(path: str) -> network.Network:
    """
    This method loads the neural network from a file
    using the pickle library functions.

    Parameters
    ----------
    path: str
        Path to file

    Returns
    -------
    net: network.Network
        Neural network to be loaded
    """
    with open(path, "rb") as file:
        net = pickle.load(file)
    return net


def import_csv_table(path: str) -> np.ndarray:
    """
    Import csv table as np array.

    Parameters
    ----------
    path: str
        Path to csv table
    Returns
    -------
    table: np.ndarray
        Parsing result
    """
    table = np.genfromtxt(path, delimiter=',')
    return table


def split_table(table: np.ndarray, len_answers=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splitting the original table into tables of variables and answers.

    Parameters
    ----------
    table: np.ndarray
        Input table
    len_answers: int
        Amount of answer variables (in columns)
    Returns
    -------
    tables: Tuple[np.ndarray, np.ndarray]
        Pair of inputs and results tables
    """
    x = table[:, :-len_answers]
    y = table[:, -len_answers:]
    return x, y


def build_plot(network: network.Network, interval: Tuple[float, float], step: float) -> None:
    """
    This method saves the neural network to a file
    using the pickle library functions.

    Parameters
    ----------
    network: network.Network
        Neural network for plotting
    interval: Tuple[float, float]
        The interval for which the plot will be built
    step: float
        Interval coverage step (number of points per interval)
    Returns
    -------
    None
    """
    x = []
    a = interval[0]
    b = interval[1]
    while a <= b:
        x.append(a)
        a += step
    y = []
    for i in x:
        temp = float(network.feedforward(i))
        y.append(temp)
    plt.plot(x, y)
    plt.show()


def shuffle_table(table: np.ndarray) -> np.ndarray:
    """
    For shuffling table.

    Parameters
    ----------
    table: np.ndarray
        Table to be shuffled
    Returns
    -------
    table: np.ndarray
        Result of shuffle
    """
    np.random.shuffle(table)
    return table
