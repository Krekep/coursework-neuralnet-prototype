import keras.activations
import keras.initializers
import numpy as np
import pytest

from networks import layer_creator
from networks.layers.dense import MyDense
from tests.utils import array_compare, init_params


@pytest.mark.parametrize(
    "inp, shape, activation, biases, expected",
    [
        (3, 10, "relu", "zeros", True),
        (1, 1, "linear", "ones", True),
        (1, 0, "relu", "zeros", False),
        (0, 1, "relu", "zeros", False),
        (1, -1, "relu", "zeros", False),
        (-1, 1, "relu", "zeros", False),
    ],
)
def test_create_layer(inp, shape, activation, biases, expected):
    act_func, bias_initializer = init_params(act_name=activation, bias_name=biases)
    try:
        _ = layer_creator.create_dense(
            inp_size=inp, shape=shape, activation=act_func, bias=bias_initializer
        )
        assert expected
    except:
        assert not expected


@pytest.mark.parametrize(
    "inp, shape, w_init, b_init, expected",
    [
        (
            np.array([[1]], dtype=float),
            [1, 1],
            "ones",
            "zeros",
            np.array([[1]], dtype=float),
        ),
        (
            np.array([[1]], dtype=float),
            [1, 1],
            "zeros",
            "zeros",
            np.array([[0]], dtype=float),
        ),
        (
            np.array([[1]], dtype=float),
            [1, 1],
            "zeros",
            "ones",
            np.array([[1]], dtype=float),
        ),
        (
            np.array([[1, 1]], dtype=float),
            [2, 1],
            "ones",
            "zeros",
            np.array([[2]], dtype=float),
        ),
        (
            np.array([[1], [1]], dtype=float),
            [1, 1],
            "ones",
            "zeros",
            np.array([[1], [1]], dtype=float),
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, 1],
            "ones",
            "zeros",
            np.array([[2], [2]], dtype=float),
        ),
    ],
)
def test_dense_call(inp, shape, w_init, b_init, expected):
    weight_initializer, bias_initializer = init_params(
        weight_name=w_init, bias_name=b_init
    )
    layer = MyDense(
        shape[0],
        shape[1],
        weight_initializer=weight_initializer,
        bias_initializer=bias_initializer,
    )

    res = layer(inp).numpy()
    assert array_compare(res, expected)


@pytest.mark.parametrize(
    "inp, shape, act_init, w_init, b_init, expected",
    [
        (
            np.array([[1]], dtype=float),
            [1, 1],
            "linear",
            "ones",
            "zeros",
            np.array([[1]], dtype=float),
        ),
        (
            np.array([[1]], dtype=float),
            [1, 1],
            "linear",
            "zeros",
            "zeros",
            np.array([[0]], dtype=float),
        ),
        (
            np.array([[1]], dtype=float),
            [1, 1],
            "linear",
            "zeros",
            "ones",
            np.array([[1]], dtype=float),
        ),
        (
            np.array([[1, 1]], dtype=float),
            [2, 1],
            "linear",
            "ones",
            "zeros",
            np.array([[2]], dtype=float),
        ),
        (
            np.array([[1], [1]], dtype=float),
            [1, 1],
            "linear",
            "ones",
            "zeros",
            np.array([[1], [1]], dtype=float),
        ),
        (
            np.array([[1, 1], [1, 1]], dtype=float),
            [2, 1],
            "linear",
            "ones",
            "zeros",
            np.array([[2], [2]], dtype=float),
        ),
    ],
)
def test_layer_call(inp, shape, act_init, w_init, b_init, expected):
    act_func, weight_initializer, bias_initializer = init_params(
        act_name=act_init, weight_name=w_init, bias_name=b_init
    )

    layer = layer_creator.create_dense(
        inp_size=shape[0],
        shape=shape[1],
        activation=act_func,
        weight=weight_initializer,
        bias=bias_initializer,
    )
    res = layer(inp).numpy()
    assert array_compare(res, expected)
