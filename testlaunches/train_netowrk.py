import datetime
import time

from project.networks.losses import MyMSE
from project.trainer import train
from project import utils
import testlaunches.build_tables
import tensorflow as tf

from openpyxl import Workbook


def load_tables(folder: str, table_name: str):
    table = utils.import_csv_table(f"./solution_tables/{folder}/{table_name}.csv")
    table = utils.shuffle_table(table)
    return utils.split_table_by_inp(table)


def calculate_mse(x: tf.Tensor, y: tf.Tensor) -> tf.Tensor:
    return tf.math.reduce_mean(tf.math.squared_difference(x, y))


names = []
for func in testlaunches.build_tables.list_sol_functions:
    names.append(func.__name__)

for func in testlaunches.build_tables.list_table_functions:
    names.append(func.__name__)
folders = ["binom", "uniform", "norm"]

data = []
for distribution in folders:
    data.append(dict())
    for func_name in names:
        x, y = load_tables(distribution, func_name)
        data[-1][func_name] = (x, y)

validation_tables = {}
for func_name in names:
    x, y = load_tables('validation_data', func_name)
    validation_tables[func_name] = (x, y)

validation_metrics = {
    "MSE": calculate_mse
}

loss_func = MyMSE()
epochs = 100
rate = 1e-2
optimizer = tf.keras.optimizers.SGD
metrics = [tf.keras.metrics.MeanAbsoluteError(),
           tf.keras.metrics.MeanSquaredLogarithmicError()]
validation_split = 0.2
normalize = False

column_names = []
for i in range(26):
    column_names.append(chr(ord('A') + i))

for i in range(26):
    column_names.append('A' + chr(ord('A') + i))

for i in range(26):
    column_names.append('B' + chr(ord('A') + i))

for i in range(26):
    column_names.append('С' + chr(ord('A') + i))

shape_separater = '-'

print("Start experiments")
for normalize in [False, True]:
    for optimizer in [tf.keras.optimizers.SGD, tf.keras.optimizers.Adam, tf.keras.optimizers.RMSprop]:
        wb = Workbook()
        wb_validate = Workbook()

        temp = str(optimizer)
        sheet_name = f"{normalize} {temp[temp.rfind('.') + 1:-2]}"
        wb.create_sheet(sheet_name)
        wb_validate.create_sheet(sheet_name)
        column_offset = 0
        column_out_offset = 0
        row_offset = 1

        for folder_num, distribution in enumerate(folders):
            for epochs in [1, 1]:
                for rate in [1e-2, 5e-3, 1e-3]:
                    loss_func_name = str(loss_func)[str(loss_func).rfind('.') + 1:str(loss_func).find(' object')]
                    wb[sheet_name][f'{column_names[column_out_offset + 1]}{row_offset}'] = f"loss = {loss_func_name}, " \
                                                                                           f"epochs = {epochs}, " \
                                                                                           f"rate = {rate}, " \
                                                                                           f"validation split = {validation_split}, " \
                                                                                           f"normalize = {normalize}"

                    wb[sheet_name][f'{column_names[column_offset]}{row_offset + 1}'] = 'loss = MSE'

                    wb_validate[sheet_name][
                        f'{column_names[column_out_offset + 1]}{row_offset}'] = f"loss = {loss_func_name}, " \
                                                                                f"epochs = {epochs}, " \
                                                                                f"rate = {rate}, " \
                                                                                f"validation split = {validation_split}, " \
                                                                                f"normalize = {normalize}"

                    wb_validate[sheet_name][f'{column_names[column_offset]}{row_offset + 1}'] = 'loss = MSE'

                    for func_name in names:
                        x, y = data[folder_num][func_name]
                        x_val, y_val = validation_tables[func_name]
                        train_start_time = time.time()
                        nets, history = train(x, y, eps=rate, epochs=epochs, validation_split=validation_split,
                                              normalize=normalize, use_rand_net=False,
                                              name_salt=func_name + shape_separater,
                                              loss_func=loss_func,
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
                        train_end_time = time.time()
                        validation_start_time = time.time()
                        for net_num, net in enumerate(nets):
                            net_shape = "_".join(map(str, net.get_shape))
                            for metric_num, val_metrics in enumerate(validation_metrics):
                                loss_val = calculate_mse(tf.convert_to_tensor(y_val, dtype=tf.float32),
                                                         net.feedforward(x_val))
                                wb[sheet_name][
                                    f'{column_names[column_out_offset]}{row_offset + 1 + metric_num * 9}'] = f'metric = {val_metrics}'
                                wb[sheet_name][
                                    f'{column_names[column_out_offset + column_offset + 1]}{row_offset + 1 + metric_num * 9}'] = f'{func_name}'
                                wb[sheet_name][
                                    f'{column_names[column_out_offset]}{row_offset + net_num + 1 + metric_num * 9 + 1}'] = f'{net_shape}'
                                wb[sheet_name][
                                    f'{column_names[column_out_offset + column_offset + 1]}{row_offset + net_num + 1 + metric_num * 9 + 1}'] = f'{loss_val}'
                        validation_end_time = time.time()
                        print(f"Train time = {train_end_time - train_start_time}. "
                              f"Validation time = {validation_end_time - validation_start_time}")
                        column_offset += 1
                    column_out_offset += column_offset + 2
                    column_offset = 0

                row_offset += 56
                column_out_offset = 0
                column_offset = 0

                print("End epochs = ", epochs)
            date = datetime.datetime.now()
            date_str = date.strftime('%I_%M_%S%p on %B %d, %Y')
            wb.save(f"{distribution}_{date_str}.xlsx")
            date_str = date.strftime('%I_%M_%S%p on %B %d, %Y')
            wb.save(f"validation_{date_str}.xlsx")
