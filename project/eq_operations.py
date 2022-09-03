import numpy as np
from math import *

from project import utils

from matplotlib import pyplot as plt


def _equation_solve(eq: str, axes: list[tuple[str, tuple[float, float, float]]]) -> list:
    """
    Supporting method for building a table with solutions of the given equation in points.

    Parameters
    ----------
    eq: str
        Neural network for build table
    axes: list[tuple[str, tuple[float, float, float]]]
        List of variables with parameters (left, right and step).
    Returns
    -------
    table: list
        Table with n+1 column, where n is the number of variables,
        the first 1 column contain the result of the expression,
        and the last n columns in each row are the values of the variables in the same order as they were input
    """
    if axes:
        curr_axis = axes.pop()
        solution_table = []
        i = curr_axis[1][0]
        while i <= curr_axis[1][1]:
            teq = eq.replace(curr_axis[0], "(" + str(i) + ")")
            res = _equation_solve(teq, axes)
            for temp in res:
                temp.append(i)
                solution_table.append(temp)
            i += curr_axis[1][2]
        axes.append(curr_axis)
        return solution_table
    else:
        eq_calc = compile(eq, "eq_compile_log.txt", "eval")
        return [[eval(eq_calc)]]


def equation_solve(eq: str, axes: list[tuple[str, tuple[float, float, float]]], debug=False) -> np.ndarray:
    """
    Method for building a table with solutions of the given equation in points.

    Parameters
    ----------
    eq: str
        Neural network for build table
    axes: list[tuple[str, tuple[float, float, float]]]
        List of variables with parameters (left, right and step).
    debug: bool
        Is debug output enabled.
    Returns
    -------
    table: np.ndarray
        Table with n+1 column, where n is the number of variables,
        the first n columns in each row are the values of the variables in the same order as they were input,
        and the last 1 column contain the result of the expression
    """
    table = _equation_solve(eq, axes)
    t = []
    for i in table:
        t.append([*i[1:], i[0]])
    res = np.array(t)
    if debug:
        utils.export_csv_table(res, "solveTable.csv")

        if len(axes) == 1:
            plt.rc("font", size=24)
            plt.figure()
            t = res[:, :1]
            y = res[:, 1:].T

            for i, y_i in enumerate(y):
                plt.plot(t, y_i, '-', label=f'{i}')
            plt.legend()
            plt.show()
    return res
