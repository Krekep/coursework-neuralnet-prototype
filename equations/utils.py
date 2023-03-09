"""
Provide some helpful functions for DE
"""

from typing import Tuple, List

import numpy as np
from matplotlib import pyplot as plt

from networks import imodel


def system_ode_from_string(system: str) -> List[List[str]]:
    s = system.split("\n")
    parsed_s = []
    for eq in s:
        parsed_s.append(eq.split())
    return parsed_s


def extract_iv(eq: str) -> Tuple[float, float]:
    """
    Extract initial value for Cauchy problem
    Format --- yi(value1)=value2 without spaces

    Parameters
    ----------
    eq: str
        Condition
    Returns
    -------
    iv: tuple[float, float]
        Pair of point and value at point
    """
    left_v = float(eq[eq.index("(") + 1 : eq.index(")")])
    right_v = float(eq[eq.index("=") + 1 :])
    return left_v, right_v


def build_plot(
    network: imodel.IModel, interval: Tuple[float, float], step: float, is_debug=False
) -> None:
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
    y = []
    for i in range(output_size):
        y.append([])

    for i in x:
        temp = network.feedforward(np.array([[i]]))
        for j in range(output_size):
            y[j].append(temp[0][j].numpy())
        if is_debug and i % (len(x) // 10) == 0:
            print("Build y data is ready ---", i % (len(x) // 10))
    if is_debug:
        print("End build y data from network")
    for i, y_i in enumerate(y):
        plt.plot(x, y_i, "-", label=f"{i}")
    plt.legend()
    plt.show()
