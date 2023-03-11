import datetime
import time

from networks import losses
from networks.trainer import train, full_search
from networks import utils
import testlaunches.build_tables
import tensorflow as tf
import csv


def load_tables(folder: str, table_name: str, input_size: int = 1):
    table = utils.import_csv_table(f"./solution_tables/{folder}/{table_name}.csv")
    table = utils.shuffle_table(table)
    return utils.split_table_by_inp(table, input_size)


names = []
for func in testlaunches.build_tables.list_sol_functions:
    names.append(func.__name__)

for func in testlaunches.build_tables.list_table_functions:
    names.append(func.__name__)


data = dict()
for func_name in names:
    x, y = load_tables('data', func_name)
    data[func_name] = (x, y)

validation_tables = {}
for func_name in names:
    x, y = load_tables('validation_data', func_name)
    validation_tables[func_name] = (x, y)

for func_name in names:
    x, y = data[func_name]
    x_val, y_val = validation_tables[func_name]
    train_start_time = time.time()
    train_results = full_search(x, y, experiments=True)
    for train_params, networks, history in train_results:
        print(train_params)
        print(networks)
        print(history)

