import numpy as np

from functions import *


def prepare_interval(interval: tuple[float, float], step: float):
    res = []
    a = interval[0]
    b = interval[1]
    while a <= b:
        res.append(a)
        a += step
    return res


list_sol_functions = [LF_ODE_1_solution, LF_ODE_2_solution, LF_ODE_3_solution, ST_LF_ODE_1_solution,
                      LH_ODE_1_solution, LH_ODE_2_solution, ST_LH_ODE_2_solution,
                      S_ODE_1_solution]

list_table_functions = [S_ODE_2_table, ST_S_ODE_3_table]

if __name__ == "__main__":
    step = 0.01
    for i in range(0, 4):
        func = list_sol_functions[i]
        x = prepare_interval((0, 1), step)
        y = [func(i) for i in x]
        table = np.array([x, y])
        table = table.transpose()
        np.savetxt(f"./solution_tables/{func.__name__}.csv", table, delimiter=",")

    step = 0.01
    for i in range(4, 6):
        func = list_sol_functions[i]
        x = prepare_interval((0.1, 1), step)
        y = [func(i) for i in x]
        table = np.array([x, y])
        table = table.transpose()
        np.savetxt(f"./solution_tables/{func.__name__}.csv", table, delimiter=",")

    step = 0.01
    for i in range(7, 8):
        func = list_sol_functions[i]
        x = prepare_interval((0, 1), step)
        y = np.array([func(i) for i in x])
        x = np.array([x]).T
        table = np.hstack([x, y])
        np.savetxt(f"./solution_tables/{func.__name__}.csv", table, delimiter=",")

    step = 0.01
    for i in range(2):
        func = list_table_functions[i]
        table = func(step)
        np.savetxt(f"./solution_tables/{func.__name__}.csv", table, delimiter=",")
