import time
from typing import Tuple

import numpy as np
from scipy import stats

from equations.system_ode import SystemODE
from networks import callbacks, trainer
from networks.config_format import HEADER_OF_FILE
from networks.imodel import IModel
import tensorflow as tf

# act_func, weight_initializer1, bias_initializer = init_params(act_name='linear', weight_name='ones',
#                                                               bias_name='zeros')
# weight_initializer2 = init_params(weight_name='zeros')[0]
# prec_act = activations.perceptron_threshold
#
# layer = ILayer(inp_size=1, shape=1, activation=act_func, weight=weight_initializer1, bias=bias_initializer)
# layer1 = MyDense(input_dim=3, units=1, activation_funcs=act_func, weight_initializer=weight_initializer1, bias_initializer=bias_initializer)
#
# print(layer.get_config())
# print(layer.get_weights())
#
# nn = IModel.create_neuron(1, 1, [1], activation=prec_act(1), weight=weight_initializer1, biases=bias_initializer, activation_names=["perceptron_threshold"], decorator_params=[1])
#
# nn1 = IModel.create_neuron(1, 1, [1], activation=prec_act(2), weight=weight_initializer2, biases=bias_initializer, activation_names=["perceptron_threshold"], decorator_params=[2])
#
# densenet = DenseNet(input_size=1, block_size=[1], output_size=1)
# print(densenet.get_config())
#
# nn.to_dict("nn.txt")
# nn1.to_dict("nn1_before.txt")
#
# nn1.from_dict("nn.txt")
#
# nn1.to_dict("nn1_after.txt")
#
# # print("**** KERAS OBJECTS ****")
# #
# # kl = keras.layers.Dense(4)
# # print(kl.get_config())
# # print(kl.get_weights())
# #
# # k = keras.Sequential()
# # k.add(keras.layers.Dense(3, input_shape=(4,)))
# # # Afterwards, we do automatic shape inference:
# # k.add(layer1)
# # print(k.get_config())
# # print(k.get_weights())
# #
# # # k.save_weights("test.h5")
# #
# # k2 = keras.Sequential()
# # k2.add(keras.layers.Dense(3, input_shape=(4,)))
# # # Afterwards, we do automatic shape inference:
# # k2.add(keras.layers.Dense(1, input_shape=(3,)))
# # print(k2.get_config())
# # print(k2.get_weights())
# # k2.load_weights("test.h5")
# # print(k2.get_weights())

# print(HEADER_OF_FILE)
# print(HEADER_OF_FILE.count("\n"))
# inp = np.array([[1]], dtype=float)
# nn = IModel(
#     1,
#     [1],
#     1,
# )
# nn.export_to_file("./test_export")
#
# nn_loaded = IModel(
#     1,
#     [1],
#     1,
# )
# nn_loaded.from_file("./test_export")
# nn_loaded.export_to_file("./test_export1")
#
# x = inp.copy()
# x1 = inp.copy()
#
# print(nn.network(inp).numpy())
# print(nn_loaded.network(inp).numpy())
#
# print("FIRST")
# for i, layer in enumerate(nn.network.blocks):
#     x = layer(x)
#     print(layer.w.numpy(), layer.b.numpy())
#     print(f"Layer {i}", x)
#
# print(
#     nn.network.out_layer.w.numpy(),
#     nn.network.out_layer.b.numpy(),
#     nn.network.out_layer.activation_func,
# )
# x = nn.network.out_layer(x)
# print(f"Classifier", x)
# print()
#
# print("SECOND")
# for i, layer in enumerate(nn_loaded.network.blocks):
#     x1 = layer(x1)
#     print(layer.w.numpy(), layer.b.numpy())
#     print(f"Layer {i}", x1)
#
# print(
#     nn_loaded.network.out_layer.w.numpy(),
#     nn_loaded.network.out_layer.b.numpy(),
#     nn_loaded.network.out_layer.activation_func,
# )
# x1 = nn_loaded.network.out_layer(x1)
# print(f"Classifier", x1)


# def f_x(x):
#     return 2 * x
#
#
# def f_x2(x):
#     return x ** 2
#
#
# def f_x_z(x, z):
#     return 2 * x - z
#
#
# x_data = np.array([[i / 10] for i in range(0, 101)])
# f_x_data = np.array([f_x2(x) for x in x_data])
#
# networks = trainer.full_search(x_data, f_x_data)
# for nn in networks:
#     # print(nn)
#     print(nn[0]["loss_func"], nn[0]["normalize"], nn[0]["epochs"], nn[0]["optimizer"], end='\n')
#     print(nn[1], nn[2], str(nn[3]))
#     print("***********")
#
# nn = IModel.create_neuron(2, 2, [2])
# nn.export_to_cpp("test")
# nn.export_to_file("test_desc")
#
# inp = np.array([[5, 5]], dtype=float)
# print(nn.feedforward(inp))
#
# acts = nn.get_activations
# acts_name = []
# for i in range(len(acts)):
#     acts_name.append(acts[i])
# print(acts_name)


# import networks.trainer
# from networks import activations, losses
# from equations.utils import build_plot
# from math import sin
# import matplotlib.pyplot as plt
#
#
# def f_x(x):
#     return 2 * x
#
#
# def f_x2(x):
#     return x ** 2
#
#
# def f_x_z(x, z):
#     return 2 * x - z
#
#
# def sin_x(x):
#     return sin(x)
#
#
# x_data = np.array([[i / 10] for i in range(0, 101)])
# f_x_data = np.array([sin_x(x) for x in x_data])
#
# shapes = [10, 10]
#
# acts = ["exponential"] * 2 + ["linear"]
#
# input_len = 1
# output_len = 1
# nets = IModel(
#     input_size=input_len,
#     block_size=shapes,
#     output_size=output_len,
#     activation_func=acts,
# )
# all_l = [key for key in losses.get_all_loss_functions()]
# for los in all_l:
#     nets.compile(optimizer="Adam", loss_func=los)
#
#     his = nets.train(
#         x_data,
#         f_x_data,
#         epochs=100,
#         verbose=0,
#     )
#     print(los, his.history["loss"][-1])
#
#     build_plot(nets, (0.0, 10.0), 0.01, title=los)
# plt.plot(x_data, f_x_data)
# plt.title("original")
# plt.show()

# import networks.trainer
# from networks import activations, losses
# from equations.utils import build_plot
# from math import sin
# import matplotlib.pyplot as plt
# from testlaunches.functions import LF_ODE_1_solution, ST_S_ODE_3_table
#
#
# interval = (0, 10)
# x_data = np.array([[i / 50] for i in range(0, 51)])
# f_x_data = np.array([LF_ODE_1_solution(x) for x in x_data])
#
# shapes = [10, 10, 10, 10, 10, 10]
#
# acts = [
#     ["swish"] * 6 + ["linear"],
#     ["relu"] * 6 + ["linear"]
# ]
#
# optimizers = ["SGD", "Adam", "RMSprop"]
#
# # los1 = "MeanSquaredError"
# los1 = "Huber"
# epochs = 200
#
# input_len = 1
# output_len = 1
#
# all_l = losses.get_all_loss_functions()
# for opt in optimizers:
#     nets = [
#         IModel(
#             input_size=input_len,
#             block_size=shapes,
#             output_size=output_len,
#             activation_func=acts[0],
#         ),
#         IModel(
#             input_size=input_len,
#             block_size=shapes,
#             output_size=output_len,
#             activation_func=acts[1],
#         )
#     ]
#     for i, nn in enumerate(nets):
#         nn.compile(optimizer=opt, loss_func=los1)
#
#         his = nn.train(
#             x_data,
#             f_x_data,
#             epochs=epochs,
#             verbose=0,
#         )
#         print(opt, his.history["loss"][-1])
#
#     build_plot(nets, (0.0, 1.0), 0.001, title=opt,
#                labels=[acts[0][0], acts[1][1], "f(x) = e^(3x)"],
#                true_data=(x_data, f_x_data))
# plt.plot(x_data, f_x_data)
# plt.title("original")
# plt.show()
#
# from random import random
#
#
# input_size = 1
# shape1 = [100, 100, 100]
# shape2 = [500, 500, 500]
# output_size = 1
#
# time_measurer = callbacks.MeasureTrainTime()
#
# nn = IModel(
#     input_size,
#     shape1,
#     output_size
# )
# nn.compile()
#
# single_data_size_call = [500, 5_000, 25_000, 50_000]
# # single_data_size_call = [500]
# single_x_data_call = [np.array([[[random() * 10]] for _ in range(0, size)]) for size in single_data_size_call]
#
# for size in single_x_data_call:
#     start_time = time.perf_counter()
#     for row in size:
#         a = nn.feedforward(row)
#     end_time = time.perf_counter()
#     call_time = end_time - start_time
#     print(f"nn1 with {shape1} have single call time {call_time} on {len(size)} size")
#
# nn.export_to_cpp("test1")
#
# nn = IModel(
#     input_size,
#     shape2,
#     output_size
# )
# nn.compile()
# single_data_size_call = [500, 5_000, 25_000, 50_000]
# single_x_data_call = [np.array([[[random() * 10]] for _ in range(0, size)]) for size in single_data_size_call]
#
# for size in single_x_data_call:
#     start_time = time.perf_counter()
#     for row in size:
#         a = nn.feedforward(row)
#     end_time = time.perf_counter()
#     call_time = end_time - start_time
#     print(f"nn2 with {shape2} have single call time {call_time} on {len(size)} size")
#
# nn.export_to_cpp("test2")
#
# import networks.trainer
# from networks import activations, losses
# from equations.utils import build_plot
# from math import sin
# import matplotlib.pyplot as plt
# from testlaunches.functions import ST_S_ODE_3_table
#
# a = 0
# b = 40
# step = 0.01
# #
# # start_time = time.perf_counter()
# # while a <= b:
# #     ST_S_ODE_3_table([a])
# #     a += step
# #
# # end_time = time.perf_counter()
# # calc_time = end_time - start_time
# print(f"scipy take {63.02206040000601} ms for {(b - a) // step} size")
#
# nn_data_x = [i / 100 for i in range(0, 4_001)]
# # nn_data_x = [i for i in range(0, 41)]
# table = ST_S_ODE_3_table(nn_data_x)
# temp = np.hsplit(table, np.array([1, 4]))
# nn_data_x = temp[0]
# nn_data_y = temp[1]
# shapes = [10, 10, 10, 10, 10, 10]
#
# acts = ["swish"] * 6 + ["linear"]
#
# # los1 = "MeanSquaredError"
# los1 = "Huber"
# epochs = 100
#
# input_len = 1
# output_len = 3
#
# nn = IModel(
#     input_size=input_len,
#     block_size=shapes,
#     output_size=output_len,
#     activation_func=acts,
# )
# opt = "Adam"
#
# nn.compile(optimizer=opt, loss_func=los1)
#
# time_measurer = callbacks.MeasureTrainTime()
# his = nn.train(
#     nn_data_x,
#     nn_data_y,
#     epochs=epochs,
#     verbose=0,
#     callbacks=[time_measurer]
# )
# print(f"nn {shapes} have train time {nn.network.trained_time['train_time']} and error is {his.history['loss'][-1]}")
#
# build_plot(nn, (0.0, 40.0), 0.02, true_data=[nn_data_x, nn_data_y])
#
# nn.export_to_cpp("timemeas")
