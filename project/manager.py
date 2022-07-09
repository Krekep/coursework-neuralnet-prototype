"""
Module representing the functions that the parser calls
"""

import sys
from argparse import Namespace
from pathlib import Path
from typing import Tuple

import project
from project import utils
from project import parser
from project import network
from project import trainer
from project import eq_operations

__all__ = [
    "create_ode_net",
    "create_table_net",
    "save_network",
    "load_network",
    "quit_comm",
]

_network_pool = {}
_count_of_nn = 0


def create_ode_net(args: Namespace) -> None:
    """
    Creating of neural network for solving ODE

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """
    pass


def _add_network_to_pool(nn, name):
    global _count_of_nn
    _network_pool[name] = nn
    _count_of_nn += 1
    print(f"Success create {name} neural network")


def print_net_info(args: Namespace) -> None:
    """
    This method prints to the console information about the neural network by its name

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """

    if args.network_name not in _network_pool.keys():
        raise Exception("No neural network exist with this name!")

    network = _network_pool[args.network_name]
    print(network)


def create_table_net(args: Namespace) -> None:
    """
    Creating of neural network for solving data table

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """
    result_path = f"{args.table_name}"
    result_file = Path(result_path)

    if not result_file.is_file():
        raise Exception(f"No such file exist {result_path}")

    table = utils.import_csv_table(result_path)
    table = utils.shuffle_table(table)
    x, y = utils.split_table(table)

    nn = trainer.train(x, y)
    nn.set_name(f"table{_count_of_nn}")
    _add_network_to_pool(nn, f"table{_count_of_nn}")


def create_equation_net(args: Namespace) -> None:
    """
    Creating of neural network for solving equation

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """
    variables = []
    for key in args.variables:
        l, r, s = map(float, args.variables[key].split(","))
        record = (key, (l, r, s))
        variables.append(record)

    table = eq_operations.equation_solve(args.equation, variables)
    x, y = utils.split_table(table)

    nn = trainer.train(x, y)
    nn.set_name(f"equation{_count_of_nn}")
    _add_network_to_pool(nn, f"equation{_count_of_nn}")


def build_plot(args: Namespace) -> None:
    """
    This method builds a graph based on the results of the neural network on the interval.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """

    network = get_network(args.network_name)
    interval = (float(args.interval[0]), float(args.interval[1]))
    step = float(args.step)
    utils.build_plot(network, interval, step)


def export_solve(args: Namespace) -> None:
    """
    This method creates a point table with equation and variable values and exports them to a file.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """

    network = get_network(args.network_name)

    result_path = f"{args.folder_path}/{args.network_name}_var_{len(args.variables)}.csv"

    variables = []
    for key in args.variables:
        l, r, s = map(float, args.variables[key].split(","))
        record = (key, (l, r, s))
        variables.append(record)
    table = utils.build_table(network, variables)
    utils.export_csv_table(table, result_path)


def save_network(args: Namespace) -> None:
    """
    This method provide use export_network with console arguments.
    Saves given network to the txt file by folder_path.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """

    network = get_network(args.network_name)

    result_path = f"{args.folder_path}/{args.network_name}.txt"

    utils.export_network(result_path, network)

    print(f"Neural network was saved in {result_path}")


def load_network(args: Namespace) -> None:
    """
    This method provide use import_network with console arguments.
    Load network from given txt file by folder_path.

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """

    result_path = f"{args.folder_path}"
    result_file = Path(result_path)

    if not result_file.is_file():
        raise Exception(f"No such file exist {result_path}")

    if args.network_name not in _network_pool.keys():
        network = utils.import_network(result_path)
        if network is not None:
            _network_pool[args.network_name] = network
        else:
            raise Exception(f"Error while loading the neural network {args.network_name}")
    else:
        raise Exception(f"Network with the name {args.network_name} already exist!")


def quit_comm(args: Namespace) -> None:
    """
    This method terminates the entire application using sys.exit(0).

    Parameters
    ----------
    args: argparse.Namespace
        Parsed arguments

    Returns
    -------
    None
    """
    print("Quit...")
    sys.exit(0)


def get_network(network_name: str) -> network.Network:
    """
    Return neural network by name

    Parameters
    ----------
    network_name: str
        Graph name

    Returns
    -------
    network.Network
        Resulting network
    """

    is_network_exist = False
    network = None

    if network_name in _network_pool.keys():
        network = _network_pool[network_name]
        is_network_exist = True

    if not is_network_exist:
        raise Exception("No such network exists!")

    return network
