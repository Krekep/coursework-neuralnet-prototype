import numpy as np
from scipy import stats
from testlaunches.functions import *
from typing import Union, List, Tuple


rng = np.random.default_rng()


# def prepare_interval(interval: Tuple[float, float], step: float, distr="uniform"):
#     a = interval[0]
#     b = interval[1]
#
#     points_count = int((b - a) / step)
#     scale = b - a
#     if distr == "uniform":
#         d = distributions[distr](loc=a, scale=b - a, size=points_count)
#     elif distr == "norm":
#         d = distributions[distr](
#             loc=a + (b - a) / 2, scale=(b - a) / 2, size=2 * points_count
#         )
#     elif distr == "binom" or distr == "nbinom":
#         d = (
#             distributions[distr](
#                 n=points_count * 30, p=0.5, loc=0, size=points_count * 2
#             )
#             / 30
#             / points_count
#             * scale
#             + a
#         )
#
#     temp = np.unique(d[(a <= d) & (d <= b)]).tolist()
#     res = sorted(temp)
#
#     return res


def prepare_uniform_interval(interval: Tuple[float, float], step: float):
    a = interval[0]
    b = interval[1]
    res = []
    while a <= b:
        res.append(a)
        a += step

    return res


list_sol_functions = [
    LF_ODE_1_solution,
    LF_ODE_2_solution,
    LF_ODE_3_solution,
    ST_LF_ODE_1_solution,
    LH_ODE_1_solution,
    LH_ODE_2_solution,
    S_ODE_1_solution,
]

list_table_functions = [S_ODE_2_table, ST_LH_ODE_2_table, ST_S_ODE_3_table]
interval_for_table_func = [(0, np.pi), (0.1, 1), (0, 40)]

if __name__ == "__main__":
    step = 0.05
    for i in range(3):
        func = list_table_functions[i]
        x = prepare_uniform_interval(interval_for_table_func[i], step)
        table = func(x)
        np.savetxt(
            f"./solution_tables/{func.__name__}.csv", table, delimiter=","
        )

    step = 0.05
    for i in range(0, 4):
        func = list_sol_functions[i]
        x = prepare_uniform_interval((0, 1), step)
        y = [func(i) for i in x]
        table = np.array([x, y])
        table = table.transpose()
        np.savetxt(
            f"./solution_tables/{func.__name__}.csv", table, delimiter=","
        )

    step = 0.05
    for i in range(4, 6):
        func = list_sol_functions[i]
        x = prepare_uniform_interval((0.1, 1), step)
        y = [func(i) for i in x]
        table = np.array([x, y])
        table = table.transpose()
        np.savetxt(
            f"./solution_tables/{func.__name__}.csv", table, delimiter=","
        )

    step = 0.05
    for i in range(6, 7):
        func = list_sol_functions[i]
        x = prepare_uniform_interval((0, 1), step)
        y = np.array([func(i) for i in x])
        x = np.array([x]).T
        table = np.hstack([x, y])
        np.savetxt(
            f"./solution_tables/{func.__name__}.csv", table, delimiter=","
        )

    # build validation data
    step = 0.001
    for i in range(3):
        func = list_table_functions[i]
        x = prepare_uniform_interval(interval_for_table_func[i], step)
        table = func(x)
        np.savetxt(
            f"./solution_tables/validation_data/{func.__name__}.csv",
            table,
            delimiter=",",
        )

    step = 0.001
    for i in range(0, 4):
        func = list_sol_functions[i]
        x = prepare_uniform_interval((0, 1), step)
        y = [func(i) for i in x]
        table = np.array([x, y])
        table = table.transpose()
        np.savetxt(
            f"./solution_tables/validation_data/{func.__name__}.csv",
            table,
            delimiter=",",
        )

    step = 0.001
    for i in range(4, 6):
        func = list_sol_functions[i]
        x = prepare_uniform_interval((0.1, 1), step)
        y = [func(i) for i in x]
        table = np.array([x, y])
        table = table.transpose()
        np.savetxt(
            f"./solution_tables/validation_data/{func.__name__}.csv",
            table,
            delimiter=",",
        )

    step = 0.001
    for i in range(6, 7):
        func = list_sol_functions[i]
        x = prepare_uniform_interval((0, 1), step)
        y = np.array([func(i) for i in x])
        x = np.array([x]).T
        table = np.hstack([x, y])
        np.savetxt(
            f"./solution_tables/validation_data/{func.__name__}.csv",
            table,
            delimiter=",",
        )
