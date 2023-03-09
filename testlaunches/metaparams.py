import tensorflow as tf

from networks.losses import *

# Distribution of data
folders = ["binom", "uniform", "norm"]

# Training algorithms
optimizers = [
    tf.keras.optimizers.SGD,
    tf.keras.optimizers.Adam,
    tf.keras.optimizers.RMSprop,
]

# Should we do data normalize before training
normalize_data = [False, True]

# Full train iteration over data
epochs_data = [[50, 200, 500, 1000], [2000]]

# Learning step
rates = [[1e-2, 5e-3, 1e-3]]

# Validation metrics
validation_metrics = {
    "RelativeError": RelativeError(),
    "InlierRatio": InlierRatio(),
    "MaxDeviation": MaxDeviation(),
    "MeanDeviation": MeanDeviation(),
}

# Loss function for training
losses_functions = [
    RelativeError(),
    MaxAbsoluteDeviation(),
    get_loss("MeanSquaredError"),
]

# Training metrics
metrics = [
    tf.keras.metrics.MeanAbsoluteError(),
    tf.keras.metrics.MeanSquaredLogarithmicError(),
]

# How much percentage of samples from data will be used for validation
split_sizes = [0.2]


# Fill metaparams
def prepare_params():
    metaparams = []
    for loss_func in losses_functions:
        for normalize in normalize_data:
            for optimizer in optimizers:
                for validation_split in split_sizes:
                    for distribution in folders:
                        for epochs in epochs_data:
                            for rate in rates:
                                metaparams.append(dict())
                                metaparams[-1]["loss_func"] = loss_func
                                metaparams[-1]["normalize"] = normalize
                                metaparams[-1]["optimizer"] = optimizer
                                metaparams[-1]["validation_split"] = validation_split
                                metaparams[-1]["distribution"] = distribution
                                metaparams[-1]["epochs"] = epochs
                                metaparams[-1]["rate"] = rate
                                metaparams[-1]["metrics"] = metrics
                                metaparams[-1][
                                    "validation_metrics"
                                ] = validation_metrics
    return metaparams


def get_loss_func_name(loss_func):
    return str(loss_func)[
        str(loss_func).rfind(".") + 1 : str(loss_func).find(" object")
    ]


def get_optimizer_name(optimizer):
    temp = str(optimizer)
    return f"{temp[temp.rfind('.') + 1:-2]}"


#
# m = prepare_params()
# for i, j in enumerate(m):
#     str_param = f"'{get_loss_func_name(j['loss_func'])}' '{j['normalize']}' '{get_optimizer_name(j['optimizer'])}' " \
#                 f"'{j['validation_split']}' '{j['distribution']}' '{j['epochs']}' '{j['rate']}' "
#     print(f"{i} : {str_param}")


"""
0 : 'RelativeError' 'False' 'SGD' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
1 : 'RelativeError' 'False' 'SGD' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
2 : 'RelativeError' 'False' 'SGD' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
3 : 'RelativeError' 'False' 'SGD' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
4 : 'RelativeError' 'False' 'SGD' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
5 : 'RelativeError' 'False' 'SGD' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
6 : 'RelativeError' 'False' 'Adam' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
7 : 'RelativeError' 'False' 'Adam' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
8 : 'RelativeError' 'False' 'Adam' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
9 : 'RelativeError' 'False' 'Adam' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
10 : 'RelativeError' 'False' 'Adam' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
11 : 'RelativeError' 'False' 'Adam' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
12 : 'RelativeError' 'False' 'RMSprop' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
13 : 'RelativeError' 'False' 'RMSprop' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
14 : 'RelativeError' 'False' 'RMSprop' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
15 : 'RelativeError' 'False' 'RMSprop' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
16 : 'RelativeError' 'False' 'RMSprop' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
17 : 'RelativeError' 'False' 'RMSprop' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
18 : 'RelativeError' 'True' 'SGD' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
19 : 'RelativeError' 'True' 'SGD' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
20 : 'RelativeError' 'True' 'SGD' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
21 : 'RelativeError' 'True' 'SGD' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
22 : 'RelativeError' 'True' 'SGD' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
23 : 'RelativeError' 'True' 'SGD' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
24 : 'RelativeError' 'True' 'Adam' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
25 : 'RelativeError' 'True' 'Adam' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
26 : 'RelativeError' 'True' 'Adam' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
27 : 'RelativeError' 'True' 'Adam' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
28 : 'RelativeError' 'True' 'Adam' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
29 : 'RelativeError' 'True' 'Adam' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
30 : 'RelativeError' 'True' 'RMSprop' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
31 : 'RelativeError' 'True' 'RMSprop' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
32 : 'RelativeError' 'True' 'RMSprop' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
33 : 'RelativeError' 'True' 'RMSprop' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
34 : 'RelativeError' 'True' 'RMSprop' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
35 : 'RelativeError' 'True' 'RMSprop' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
36 : 'MaxAbsoluteDeviation' 'False' 'SGD' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
37 : 'MaxAbsoluteDeviation' 'False' 'SGD' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
38 : 'MaxAbsoluteDeviation' 'False' 'SGD' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
39 : 'MaxAbsoluteDeviation' 'False' 'SGD' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
40 : 'MaxAbsoluteDeviation' 'False' 'SGD' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
41 : 'MaxAbsoluteDeviation' 'False' 'SGD' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
42 : 'MaxAbsoluteDeviation' 'False' 'Adam' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
43 : 'MaxAbsoluteDeviation' 'False' 'Adam' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
44 : 'MaxAbsoluteDeviation' 'False' 'Adam' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
45 : 'MaxAbsoluteDeviation' 'False' 'Adam' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
46 : 'MaxAbsoluteDeviation' 'False' 'Adam' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
47 : 'MaxAbsoluteDeviation' 'False' 'Adam' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
48 : 'MaxAbsoluteDeviation' 'False' 'RMSprop' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
49 : 'MaxAbsoluteDeviation' 'False' 'RMSprop' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
50 : 'MaxAbsoluteDeviation' 'False' 'RMSprop' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
51 : 'MaxAbsoluteDeviation' 'False' 'RMSprop' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
52 : 'MaxAbsoluteDeviation' 'False' 'RMSprop' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
53 : 'MaxAbsoluteDeviation' 'False' 'RMSprop' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
54 : 'MaxAbsoluteDeviation' 'True' 'SGD' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
55 : 'MaxAbsoluteDeviation' 'True' 'SGD' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
56 : 'MaxAbsoluteDeviation' 'True' 'SGD' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
57 : 'MaxAbsoluteDeviation' 'True' 'SGD' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
58 : 'MaxAbsoluteDeviation' 'True' 'SGD' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
59 : 'MaxAbsoluteDeviation' 'True' 'SGD' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
60 : 'MaxAbsoluteDeviation' 'True' 'Adam' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
61 : 'MaxAbsoluteDeviation' 'True' 'Adam' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
62 : 'MaxAbsoluteDeviation' 'True' 'Adam' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
63 : 'MaxAbsoluteDeviation' 'True' 'Adam' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
64 : 'MaxAbsoluteDeviation' 'True' 'Adam' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
65 : 'MaxAbsoluteDeviation' 'True' 'Adam' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
66 : 'MaxAbsoluteDeviation' 'True' 'RMSprop' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
67 : 'MaxAbsoluteDeviation' 'True' 'RMSprop' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
68 : 'MaxAbsoluteDeviation' 'True' 'RMSprop' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
69 : 'MaxAbsoluteDeviation' 'True' 'RMSprop' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
70 : 'MaxAbsoluteDeviation' 'True' 'RMSprop' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
71 : 'MaxAbsoluteDeviation' 'True' 'RMSprop' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
72 : 'MyMSE' 'False' 'SGD' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
73 : 'MyMSE' 'False' 'SGD' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
74 : 'MyMSE' 'False' 'SGD' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
75 : 'MyMSE' 'False' 'SGD' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
76 : 'MyMSE' 'False' 'SGD' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
77 : 'MyMSE' 'False' 'SGD' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
78 : 'MyMSE' 'False' 'Adam' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
79 : 'MyMSE' 'False' 'Adam' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
80 : 'MyMSE' 'False' 'Adam' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
81 : 'MyMSE' 'False' 'Adam' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
82 : 'MyMSE' 'False' 'Adam' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
83 : 'MyMSE' 'False' 'Adam' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
84 : 'MyMSE' 'False' 'RMSprop' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
85 : 'MyMSE' 'False' 'RMSprop' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
86 : 'MyMSE' 'False' 'RMSprop' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
87 : 'MyMSE' 'False' 'RMSprop' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
88 : 'MyMSE' 'False' 'RMSprop' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
89 : 'MyMSE' 'False' 'RMSprop' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
90 : 'MyMSE' 'True' 'SGD' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
91 : 'MyMSE' 'True' 'SGD' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
92 : 'MyMSE' 'True' 'SGD' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
93 : 'MyMSE' 'True' 'SGD' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
94 : 'MyMSE' 'True' 'SGD' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
95 : 'MyMSE' 'True' 'SGD' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
96 : 'MyMSE' 'True' 'Adam' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
97 : 'MyMSE' 'True' 'Adam' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
98 : 'MyMSE' 'True' 'Adam' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
99 : 'MyMSE' 'True' 'Adam' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
100 : 'MyMSE' 'True' 'Adam' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
101 : 'MyMSE' 'True' 'Adam' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
102 : 'MyMSE' 'True' 'RMSprop' '0.2' 'binom' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
103 : 'MyMSE' 'True' 'RMSprop' '0.2' 'binom' '[2000]' '[0.01, 0.005, 0.001]'
104 : 'MyMSE' 'True' 'RMSprop' '0.2' 'uniform' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
105 : 'MyMSE' 'True' 'RMSprop' '0.2' 'uniform' '[2000]' '[0.01, 0.005, 0.001]'
106 : 'MyMSE' 'True' 'RMSprop' '0.2' 'norm' '[50, 200, 500, 1000]' '[0.01, 0.005, 0.001]'
107 : 'MyMSE' 'True' 'RMSprop' '0.2' 'norm' '[2000]' '[0.01, 0.005, 0.001]'
"""
