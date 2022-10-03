import datetime

from project.networks.losses import MyMSE
from project.trainer import train
from project import utils
import build_tables
import tensorflow as tf

from openpyxl import Workbook

names = []
for func in build_tables.list_sol_functions:
    names.append(func.__name__)

for func in build_tables.list_table_functions:
    names.append(func.__name__)
"""
args = {
    "debug": False,
    "eps": 1e-2,
    "epochs": 100,
    "validation_split": 0.2,
    "normalize": False,
    "name_salt": "",
    "loss_func": MyMSE(),
    "optimizer": tf.keras.optimizers.SGD,
    "metrics": [tf.keras.metrics.MeanSquaredError(), tf.keras.metrics.MeanAbsoluteError(),
                tf.keras.metrics.CosineSimilarity()]
}
"""

loss_func = MyMSE()
epochs = 100
rate = 1e-2
optimizer = tf.keras.optimizers.SGD
metrics = [tf.keras.metrics.MeanAbsoluteError(),
           tf.keras.metrics.MeanSquaredLogarithmicError()]
validation_split = 0.2
normalize = False

wb = Workbook()

column_names = []
for i in range(26):
    column_names.append(chr(ord('A') + i))

for i in range(26):
    column_names.append('A' + chr(ord('A') + i))

for i in range(26):
    column_names.append('B' + chr(ord('A') + i))

shape_separater = '-'

for normalize in [False, True]:
    for optimizer in [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop]:
        # for optimizer in [tf.keras.optimizers.SGD]:
        temp = str(optimizer)
        sheet_name = f"{normalize} {temp[temp.rfind('.') + 1:-2]}"
        wb.create_sheet(sheet_name)
        column_offset = 0
        column_out_offset = 0
        row_offset = 1

        for epochs in [50, 200, 500, 1000, 2000]:
            for rate in [1e-2, 2e-2, 5e-3]:
                loss_func_name = str(loss_func)[str(loss_func).rfind('.') + 1:str(loss_func).find(' object')]
                wb[sheet_name][f'{column_names[column_out_offset + 1]}{row_offset}'] = f"loss = {loss_func_name}, " \
                                                                                       f"epochs = {epochs}, " \
                                                                                       f"rate = {rate}, " \
                                                                                       f"validation split = {validation_split}, " \
                                                                                       f"normalize = {normalize}"

                wb[sheet_name][f'{column_names[column_offset]}{row_offset + 1}'] = 'loss = MSE'

                for func_name in names:
                    table = utils.import_csv_table(f"./solution_tables/{func_name}.csv")
                    table = utils.shuffle_table(table)
                    x, y = utils.split_table_by_inp(table)
                    for i in range(1):
                        _, history = train(x, y, eps=rate, epochs=epochs, validation_split=validation_split,
                                           normalize=normalize, use_rand_net=False,
                                           name_salt=str(i) + "_" + func_name + shape_separater, loss_func=loss_func,
                                           optimizer=optimizer, metrics=metrics, experiments=True)
                        for net_num, net_history in enumerate(history):
                            net_shape = f"{net_history.model.name[net_history.model.name.rfind(shape_separater):]}"
                            for metric_num, train_metrics in enumerate(net_history.history):
                                wb[sheet_name][
                                    f'{column_names[column_out_offset]}{row_offset + 1 + metric_num * 9}'] = f'metric = {train_metrics}'
                                wb[sheet_name][
                                    f'{column_names[column_out_offset + column_offset + 1]}{row_offset + 1 + metric_num * 9}'] = f'{func_name}'
                                wb[sheet_name][
                                    f'{column_names[column_out_offset]}{row_offset + net_num + 1 + metric_num * 9 + 1}'] = f'{net_shape[2:]}'
                                wb[sheet_name][
                                    f'{column_names[column_out_offset + column_offset + 1]}{row_offset + net_num + 1 + metric_num * 9 + 1}'] = f'{net_history.history[train_metrics][-1]}'
                    column_offset += 1
                column_out_offset += column_offset + 2
                column_offset = 0
                row_offset = 1

date = datetime.datetime.now()
date_str = date.strftime('%I_%M%p on %B %d, %Y')
wb.save(f"{date_str}.xlsx")
