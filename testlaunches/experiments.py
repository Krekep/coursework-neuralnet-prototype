import datetime
import time

import keras.backend
import numpy as np
import pandas as pd

from networks.trainer import full_search
from networks import utils
import csv
from memory_profiler import profile


def load_tables(folder: str, table_name: str, input_size: int = 1):
    table = utils.import_csv_table(f"./solution_tables/{folder}/{table_name}.csv")
    table = utils.shuffle_table(table)
    return utils.split_table_by_inp(table, input_size)


def prepare_tables(names: list[str], folder: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    data = dict()
    for func_name in names:
        x, y = load_tables(folder, func_name)
        data[func_name] = (x, y)

    return data


def do_experiments(
        names: list[str],
        data: dict[str, tuple[np.ndarray, np.ndarray]],
        val_data: dict[str, tuple[np.ndarray, np.ndarray]],
        **kwargs
):
    for fun_num, func_name in enumerate(names):
        if fun_num != 2:
            continue
        x, y = data[func_name]
        x_val, y_val = val_data[func_name]
        train_results = full_search(x, y, experiments=True, **kwargs)

        env_params = dict()
        nn_params = dict()
        history_params = dict()

        for key in train_results[0][0]:
            env_params[key] = []

        nn_params["shape"] = []
        nn_params["activations"] = []

        # for hist_key in train_results[0][2][0].history:
        #     history_params[hist_key] = []

        for hist_key in train_results[0][2][0]:
            history_params[hist_key] = []

        for train_params in train_results:
            expected_size = len(train_params[1])

            for key in train_params[0]:
                val = train_params[0][key]
                if key in ["loss_func", "optimizer", "metrics", "validation_metrics"]:
                    if key in ["metrics", "validation_metrics"]:
                        for i in range(len(val)):
                            val[i] = type(val[i]).__name__
                    elif key in ["optimizer"]:
                        # TODO: Why "type" for optimizer return "type"?
                        val = str(val)
                        dot_idx = val.rfind(".") + 1
                        val = val[dot_idx:-2]
                    else:
                        val = type(val).__name__
                env_params[key] += [val] * expected_size

            for nn in train_params[1]:
                nn_params["shape"].append(str([nn.get_input_size] + nn.get_shape + [nn.get_output_size]))
                acts = nn.get_activations
                for i in range(len(acts)):
                    acts[i] = acts[i].__name__
                nn_params["activations"].append(str(acts))

            for hist in train_params[2]:
                for hist_key in hist:
                    history_params[hist_key].append(hist[hist_key])

            # for hist in train_params[2]:
            #     for hist_key in hist.history:
            #         history_params[hist_key].append(hist.history[hist_key][-1])

        nn_params.update(env_params)
        nn_params.update(history_params)

        nn_params.pop("nets_param")
        nn_params.pop("metrics")
        nn_params.pop("validation_metrics")

        # df = pd.DataFrame(data=nn_params)
        # df = df.drop(columns=["nets_param", "metrics", "validation_metrics"])
        # print(df)
        print(func_name)
        keras.backend.clear_session()
        with open(f"./solution_tables/train_result/{func_name}.csv", "w", newline='') as outfile:
            writer = csv.writer(outfile)
            writer.writerow(nn_params.keys())
            writer.writerows(zip(*nn_params.values()))
