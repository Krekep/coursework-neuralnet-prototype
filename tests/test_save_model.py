import pytest
import numpy as np

import networks.activations
import tests.utils
from networks.imodel import IModel
from tests.utils import array_compare, init_params, file_compare


@pytest.mark.parametrize(
    "inp, shape, act_init, decorator_params",
    [
        (np.array([[1]], dtype=float), [1, [1], 1], ["sigmoid", "linear"], None),
        (np.array([[1]], dtype=float), [1, [1], 1], "sigmoid", None),
        (np.array([[1]], dtype=float), [1, [1], 1], ["linear", "linear"], None),
        (np.array([[1, 1]], dtype=float), [2, [1], 1], "tanh", None),
        (np.array([[1], [1]], dtype=float), [1, [1], 1], "tanh", None),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1], 2],
            ["perceptron_threshold"],
            [1],
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1], 1],
            ["perceptron_threshold"],
            [2],
        ),
    ],
)
def test_predict_is_same(inp, shape, act_init, decorator_params):
    act_func = tests.utils.init_params(act_name=act_init)
    act_names = [act_init] if not isinstance(act_init, list) else act_init
    if isinstance(act_func, list):
        for i in range(len(act_func)):
            act_names.append(act_init[i])
            if act_init[i] in networks.activations._decorated_activation:
                act_func[i] = act_func[i](decorator_params)
    nn = IModel(
        shape[0],
        shape[1],
        shape[2],
        activation_func=act_func,
        activation_names=act_names,
        decorator_params=decorator_params,
    )

    expected = nn.feedforward(inp).numpy()
    nn.export_to_file("./data/test_export")

    nn_loaded = IModel(
        shape[0],
        shape[1],
        shape[2],
    )
    nn_loaded.from_file("./data/test_export")
    nn_loaded.export_to_file("./data/test_export1")
    actual = nn_loaded.feedforward(inp).numpy()

    assert array_compare(actual, expected)


@pytest.mark.parametrize(
    "inp, shape",
    [
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1, 1]], dtype=float),
            [2, [1], 1],
        ),
        (
            np.array([[1], [1]], dtype=float),
            [1, [1], 1],
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1], 2],
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, [1], 1],
        ),
    ],
)
def test_file_is_same(inp, shape):
    nn = IModel(
        shape[0],
        shape[1],
        shape[2],
    )
    nn.export_to_file("./data/test_export")

    nn_loaded = IModel(
        shape[0],
        shape[1],
        shape[2],
    )
    nn_loaded.from_file("./data/test_export")
    nn_loaded.export_to_file("./data/test_export1")

    assert file_compare("./data/test_export.apg", "./data/test_export1.apg")
