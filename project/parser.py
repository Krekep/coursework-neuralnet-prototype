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


class StoreDictKeyPair(argparse.Action):
    def __init__(self, option_strings, dest, nargs=None, **kwargs):
        self._nargs = nargs
        super(StoreDictKeyPair, self).__init__(option_strings, dest, nargs=nargs, **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = {}
        # print("values: {}".format(values))
        for kv in values:
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


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
    subparsers = parser.add_subparsers(title="differential equation arguments", dest="subparser_name")

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

    # input system ODE
    parser_system_ode = subparsers.add_parser(
        "system-ode", help="Create neural net for system of ODEs. "
                           "Input format --- *equation* *initial value*"
    )
    parser_system_ode.add_argument(
        "n", type=int, help="Amount of equations"
    )
    parser_system_ode.add_argument(
        "--interval",
        metavar=("L", "R"),
        dest="interval",
        default=[0, 1],
        type=int,
        help="Interval of calculations [a, b]",
        nargs=2
    )
    parser_system_ode.add_argument(
        "--points",
        dest="points",
        default=10,
        type=int,
        help="Amount of points per interval",
    )
    parser_system_ode.set_defaults(func=manager.create_system_ode_net)

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
    parser_table.add_argument(
        "--ans",
        dest="ans_len",
        default=1,
        help="Length of answer vector for input vector (count of functions)",
    )
    parser_table.set_defaults(func=manager.create_table_net)

    # input equation
    parser_equation = subparsers.add_parser(
        "eq",
        help="Create neural net for solve equation."
    )
    parser_equation.add_argument(
        "equation",
        help="Input equation for solve"
    )
    parser_equation.add_argument(
        "--vars",
        dest="variables",
        action=StoreDictKeyPair,
        help="Variables with settings --- interval [left, right] and step. Example << x=0,5,0.2 >>",
        nargs="+",
        metavar="KEY=VAL"
    )
    parser_equation.add_argument(
        "--type",
        dest="de_type",
        default=None,
        help="Type of DE for this equation",
    )
    parser_equation.set_defaults(func=manager.create_equation_net)

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
        type=float,
        default=[0, 1],
        help="Interval of calculations [a, b]",
        nargs=2
    )
    parser_build_plot.add_argument(
        "--step",
        metavar="step",
        dest="step",
        type=float,
        default=0.01,
        help="Function value calculation step",
    )
    parser_build_plot.set_defaults(func=manager.build_plot)

    # export-solve
    parser_export_solution_table = subparsers.add_parser("export-solve", help="Export solution table.")
    parser_export_solution_table.add_argument(
        "network_name", metavar="network-name", help="Name of neural network"
    )
    parser_export_solution_table.add_argument(
        "folder_path", metavar="folder-path", help="path to folder"
    )
    parser_export_solution_table.add_argument(
        "--vars",
        dest="variables",
        action=StoreDictKeyPair,
        help="Variables with settings --- interval [left, right] and step. Example << x=0,5,0.2 >>",
        nargs="+",
        metavar="KEY=VAL"
    )
    parser_export_solution_table.set_defaults(func=manager.export_solve)

    # debug
    parser_debug = subparsers.add_parser("debug", help="Enable/disable debug output.")
    parser_debug.add_argument(
        "flag",
        type=bool,
        help="True --- enable debug output. False --- disable debug output",
    )
    parser_debug.set_defaults(func=manager.set_debug)

    # quit
    parser_quit = subparsers.add_parser("quit", help="Stop executable.")
    parser_quit.set_defaults(func=manager.quit_comm)

    return parser


def parse_ode(args: Namespace):
    pass


def parse_table(args: Namespace):
    pass
