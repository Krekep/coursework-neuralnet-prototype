from typing import Tuple

import numpy as np
from scipy import stats

from equations.system_ode import SystemODE
from networks.config_format import HEADER_OF_FILE
from networks.imodel import IModel
import tensorflow as tf

# act_func, weight_initializer1, bias_initializer = init_params(act_name='linear', weight_name='ones',
#                                                               bias_name='zeros')
# weight_initializer2 = init_params(weight_name='zeros')[0]
# prec_act = activations.perceptron_threshold
#
# layer = ILayer(inp_size=1, shape=1, activation=act_func, weight=weight_initializer1, bias=bias_initializer)
# layer1 = MyDense(input_dim=3, units=1, activation_func=act_func, weight_initializer=weight_initializer1, bias_initializer=bias_initializer)
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
#     nn.network.classifier.w.numpy(),
#     nn.network.classifier.b.numpy(),
#     nn.network.classifier.activation_func.__name__,
# )
# x = nn.network.classifier(x)
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
#     nn_loaded.network.classifier.w.numpy(),
#     nn_loaded.network.classifier.b.numpy(),
#     nn_loaded.network.classifier.activation_func.__name__,
# )
# x1 = nn_loaded.network.classifier(x1)
# print(f"Classifier", x1)
#
# def S_ODE_2_table(points_array: list, interval: Tuple[float, float] = (0, np.pi)):
#     """
#     y0' = y1 * y2 y0(0)=0
#     y1' = -y0 * y2 y1(0)=0
#     y2' = -0.5 * y0 * y1 y2(0) = 0
#     """
#     size = 3
#     sode = "y1*y2 y0(0)=0\n" \
#            "-y0*y2 y1(0)=1\n" \
#            "-0.5*y0*y1 y2(0)=1"
#     temp = sode.split("\n")
#     prepared_sode = []
#     for eq in temp:
#         prepared_sode.append(eq.split(" "))
#
#     solver = SystemODE(debug=True)
#     solver.prepare_equations(size, prepared_sode)
#     solver.solve(interval, points_array)
#     res_table = solver.build_table()
#     return res_table
#
#
# def prepare_interval(interval: Tuple[float, float], step: float, distr="uniform"):
#     a = interval[0]
#     b = interval[1]
#
#     points_count = int((b - a) / step)
#     scale = b - a
#     if distr == "uniform":
#         d = stats.uniform.rvs(loc=a, scale=b - a, size=points_count)
#
#     temp = np.unique(d[(a <= d) & (d <= b)]).tolist()
#     res = sorted(temp)
#
#     return res
#
#
# # interval_for_table_func = [(0, np.pi), (0.1, 1), (0, 40)]
# # step = 0.05
# # x = prepare_interval(interval_for_table_func[0], step)
# # table = S_ODE_2_table(x)
#
# s = """y0*2 y0(0)=1"""
# s = s.split("\n")
# parsed_s = []
# for eq in s:
#     parsed_s.append(eq.split())
#
# e = SystemODE(debug=True)
# e.prepare_equations(1, parsed_s)
# e.solve((0, 1))
# res = e.build_table([0])

# import trainer
# from networks import activations, losses
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
# x_data = np.array([[i / 10] for i in range(0, 101)])
# f_x_data = np.array([f_x(x) for x in x_data])
#
# size = 4
# shapes = [
#     [],
#     [10, 10],
#     [1, 1, 1],
#     [2]
# ]
#
# acts = [
#     [activations.get("linear")],
#     [activations.get("perceptron_threshold")] * 2 + [activations.get("linear")],
#     [activations.get("perceptron_threshold")] * 3 + [activations.get("linear")],
#     [activations.get("perceptron_threshold")] * 1 + [activations.get("linear")],
# ]
# names = [
#     ["a"],
#     ["a"] * 3,
#     ["a"] * 4,
#     ["a"] * 2,
# ]
#
# nets = []
# input_len = 1
# output_len = 1
# losses_f = losses.get_all_loss_functions()
#
# for l in losses_f:
#     for s, a, n in zip(shapes, acts, names):
#         nets.append(
#             IModel(
#                 input_size=input_len,
#                 block_size=s,
#                 output_size=output_len,
#                 activation_func=a,
#                 activation_names=n,
#             )
#         )
#
# for i, l in enumerate(losses_f):
#     for j in range(size):
#         nn = nets[i * size + j]
#         nn.compile(loss_func=losses_f[l])
#
# for nn in nets:
#     his = nn.train(
#         x_data,
#         f_x_data,
#         epochs=20,
#         verbose=0,
#     )
#     print(his.history["loss"][-1])

# networks = trainer.full_search(x_data, f_x_data)
# for nn in networks:
#     # print(nn)
#     print(nn[0]["loss_func"], nn[0]["normalize"], nn[0]["epochs"], nn[0]["optimizer"], end='\n')
#     print(nn[1], nn[2], str(nn[3]))
#     print("***********")

nn = IModel.create_neuron(2, 2, [2])
nn.export_to_cpp("test")
nn.export_to_file("test_desc")

inp = np.array([[5, 5]], dtype=float)
print(nn.feedforward(inp))

acts = nn.get_activations
acts_name = []
for i in range(len(acts)):
    acts_name.append(acts[i].__name__)
print(acts_name)
