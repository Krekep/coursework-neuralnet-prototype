import numpy as np
from tensorflow import keras

import networks.activations


def array_compare(a, b):
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if isinstance(a[i], (list, np.ndarray)):
            if isinstance(b[i], (list, np.ndarray)):
                return array_compare(a[i], b[i])
            else:
                return False
        else:
            if isinstance(b[i], (list, np.ndarray)):
                return False
            elif a[i] != b[i]:
                return False
    return True


def init_params(act_name: str = None, weight_name: str = None, bias_name: str = None):
    return_param = []
    if act_name is not None:
        if isinstance(act_name, list):
            act_func = []
            for name in act_name:
                act_func.append(networks.activations.get(name))
        else:
            act_func = networks.activations.get(act_name)
        return_param.append(act_func)
    if weight_name is not None:
        weight_initializer = keras.initializers.get(weight_name)
        return_param.append(weight_initializer)
    if bias_name is not None:
        bias_initializer = keras.initializers.get(bias_name)
        return_param.append(bias_initializer)

    if len(return_param) == 0:
        return None
    if len(return_param) == 1:
        return return_param[0]
    return return_param


def file_compare(path1: str, path2: str) -> bool:
    f1 = open(path1)
    f2 = open(path2)
    f1_lines = f1.readlines()
    f2_lines = f2.readlines()

    if len(f1_lines) != len(f2_lines):
        return False
    for i in range(len(f1_lines)):
        if not f1_lines[i].__eq__(f2_lines[i]):
            return False
    return True
