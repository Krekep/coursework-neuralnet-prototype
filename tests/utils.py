import numpy as np
from tensorflow import keras


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
        act_func = keras.activations.get(act_name)
        return_param.append(act_func)
    if weight_name is not None:
        weight_initializer = keras.initializers.get(weight_name)
        return_param.append(weight_initializer)
    if bias_name is not None:
        bias_initializer = keras.initializers.get(bias_name)
        return_param.append(bias_initializer)
    return return_param
