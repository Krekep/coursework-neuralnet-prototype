"""
Provide some helpful functions
"""

from typing import Tuple, List

from project.networks import inetwork

import pickle
import numpy as np
import matplotlib.pyplot as plt


def export_network(path: str, net: inetwork.INetwork) -> None:
    """
    This method saves the neural network to a file
    using the pickle library functions.

    Parameters
    ----------
    path: str
        Path to file
    net: network.INetwork
        Neural network to be saved
    Returns
    -------
    None
    """
    with open(path, "wb") as file:
        pickle.dump(net, file)


def import_network(path: str) -> inetwork.INetwork:
    """
    This method loads the neural network from a file
    using the pickle library functions.

    Parameters
    ----------
    path: str
        Path to file

    Returns
    -------
    net: network.INetwork
        Neural network to be loaded
    """
    with open(path, "rb") as file:
        net = pickle.load(file)
    return net


def import_csv_table(path: str) -> np.ndarray:
    """
    Import csv table as numpy array.

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


def export_csv_table(table: np.ndarray, path: str) -> None:
    """
    Export numpy array to csv table.

    Parameters
    ----------
    table: np.ndarray
        Table to export
    path: str
        Path to csv table
    Returns
    -------
    None
    """
    np.savetxt(path, table, delimiter=",")


def split_table_by_ans(table: np.ndarray, len_answers=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splitting the original table into tables of variables and answers by length of answer.

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


def split_table_by_inp(table: np.ndarray, len_input=1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Splitting the original table into tables of variables and answers by length of input.

    Parameters
    ----------
    table: np.ndarray
        Input table
    len_input: int
        Amount of input variables (in columns)
    Returns
    -------
    tables: Tuple[np.ndarray, np.ndarray]
        Pair of inputs and results tables
    """
    x = table[:, :len_input]
    y = table[:, len_input:]
    return x, y


def build_plot(network: inetwork.INetwork, interval: Tuple[float, float], step: float, is_debug=False) -> None:
    """
    Builds a two-dimensional graph on an interval with a given step.

    Parameters
    ----------
    network: network.INetwork
        Neural network for plotting
    interval: Tuple[float, float]
        The interval for which the plot will be built
    step: float
        Interval coverage step (number of points per interval)
    is_debug: bool
        Console debug output marker
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
    if is_debug:
        print("End build x data")
    output_size = network.get_output_size
    y = [0] * output_size
    for i in range(output_size):
        y[i] = []

    for i in x:
        temp = network.feedforward(np.array([[i]]))
        for j in range(output_size):
            y[j].append(temp[0][j].numpy())
        if i % (len(x) // 10) == 0 and is_debug:
            print("Build y data is ready ---", i % (len(x) // 10))
    if is_debug:
        print("End build y data from network")
    for i, y_i in enumerate(y):
        plt.plot(x, y_i, '-', label=f'{i}')
    plt.legend()
    plt.show()


def _build_table(network: inetwork.INetwork, axes: List[Tuple[str, Tuple[float, float, float]]], acc=None) -> List:
    """
    Supporting method for taken network answer.

    Parameters
    ----------
    network: network.INetwork
        Neural network for build table
    axes: list[tuple[str, tuple[float, float, float]]]
        List of variables with parameters (left, right and step).
    acc: np.ndarray
        Supporting array for variables value
    Returns
    -------
    table: list
        Table with n+k column, where n is the number of variables,
        the first k columns contain the response of the neural network,
        and the last n columns in each row are the values of the variables in the same order as they were input
    """
    if axes:
        if acc is None:
            acc = np.array([])
        curr_axis = axes.pop()
        solution_table = []
        i = curr_axis[1][0]
        while i <= curr_axis[1][1]:
            tacc = np.append(acc, [i])
            res = _build_table(network, axes, tacc)
            for temp in res:
                temp.append(i)
                solution_table.append(temp)
            i += curr_axis[1][2]
        axes.append(curr_axis)
        return solution_table
    elif acc is not None:
        temp = network.feedforward(acc[:, np.newaxis])
        res = temp[0].numpy().tolist()
        return [res]


def build_table(network: inetwork.INetwork, axes: List[Tuple[str, Tuple[float, float, float]]]) -> List:
    """
    Builds a solution table on the interval given for each variable with the given step.

    Parameters
    ----------
    network: network.INetwork
        Neural network for build table
    axes: list[tuple[str, tuple[float, float, float]]]
        List of variables with parameters (left, right and step).
    Returns
    -------
    table: list
        Table with n+k column, where n is the number of variables,
        the first n columns in each row are the values of the variables in the same order as they were input,
        and the last k columns contain the response of the neural network
    """
    table = _build_table(network, axes)
    res = []
    k = network.get_output_size
    for i in table:
        res.append([*i[k:], *i[:k]])
    return res


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
