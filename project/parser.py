"""
Parsers module
"""

import argparse
from argparse import Namespace

import project.manager
from project import manager

__all__ = [
    "parser_initialize",
    "parse_ode",
    "parse_table"
]


def parser_initialize() -> argparse.ArgumentParser:
    """
    Parses console input when starting a module

    Returns
    -------
    argparse.ArgumentParser
        Commands parser
    """
    parser = argparse.ArgumentParser(prog="python -m project")
    parser.set_defaults(func=lambda args: parser.error("too few arguments"))
    subparsers = parser.add_subparsers(title="differential equation arguments", dest="")

    # input ODE
    parser_ode = subparsers.add_parser(
        "ode", help="Create neural net for ode."
    )
    parser_ode.add_argument(
        "formula", metavar="formula", help="ode formula: y' - f(x) = 0"
    )
    parser_ode.add_argument(
        "--interval",
        metavar=("L", "R"),
        dest="interval",
        default=[0, 1],
        help="interval of calculations [a, b]",
        nargs=2
    )
    parser_ode.add_argument(
        "left_value",
        metavar="left-value",
        help="y value in the 'a' point",
    )
    parser_ode.set_defaults(func=manager.create_ode_net)

    # input table of values
    parser_table = subparsers.add_parser(
        "table",
        help="Create neural net for solve table."
    )
    parser_table.add_argument(
        "table_name",
        metavar="table-name",
        help="Path to csv table"
    )
    parser_table.add_argument(
        "--type",
        dest="de_type",
        default=None,
        help="Type of DE in table",
    )
    parser_table.set_defaults(func=manager.create_table_net)

    # save-to-file
    parser_save_to_file = subparsers.add_parser(
        "save-to-file", help="Save network to file."
    )
    parser_save_to_file.add_argument(
        "network_name", metavar="net-name", help="name of desired network"
    )
    parser_save_to_file.add_argument(
        "folder_path", metavar="folder-path", help="path to folder"
    )
    parser_save_to_file.set_defaults(func=manager.save_network)

    # load-from-file
    parser_load_from_file = subparsers.add_parser(
        "load-from-file", help="Load network from file."
    )
    parser_load_from_file.add_argument(
        "network_name", metavar="net-name", help="name of desired network"
    )
    parser_load_from_file.add_argument(
        "folder_path", metavar="folder-path", help="path to folder"
    )
    parser_load_from_file.set_defaults(func=manager.load_network)

    # print-info
    parser_print_info = subparsers.add_parser("print-info", help="Print information about neural network by name.")
    parser_print_info.add_argument(
        "network_name", metavar="network-name", help="Name of neural network"
    )
    parser_print_info.set_defaults(func=manager.print_net_info)

    # build-plot
    parser_build_plot = subparsers.add_parser("build-plot", help="Build plot on interval.")
    parser_build_plot.add_argument(
        "network_name", metavar="network-name", help="Name of neural network"
    )
    parser_build_plot.add_argument(
        "--interval",
        metavar=("L", "R"),
        dest="interval",
        default=[0, 1],
        help="Interval of calculations [a, b]",
        nargs=2
    )
    parser_build_plot.add_argument(
        "--step",
        metavar="step",
        dest="step",
        default=0.01,
        help="Function value calculation step",
    )
    parser_build_plot.set_defaults(func=manager.build_plot)

    # quit
    parser_quit = subparsers.add_parser("quit", help="Stop executable.")
    parser_quit.set_defaults(func=manager.quit_comm)

    return parser


def parse_ode(args: Namespace):
    pass


def parse_table(args: Namespace):
    pass
