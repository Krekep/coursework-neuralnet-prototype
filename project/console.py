from typing import List

from project import parser

import argparse

__all__ = ["run_console"]

_parser = parser.parser_initialize()


def parse_ode_equations(n: int) -> list[list[str]]:
    """
    Parse input string to pair of equations and initial value

    Parameters
    ----------
    n: int
        Count of equations
    Returns
    -------
    equations: list[list[str]]
        Parsed input
    """
    res = []
    for i in range(n):
        res.append(input().split())
    return res


def run_console() -> None:
    """
    Runs a console application.

    Returns
    -------
    None
    """
    while True:
        stream = input(">>> ")
        try:
            args = _parser.parse_args(stream.split())
            if args.subparser_name == "system-ode":
                additional_arguments = parse_ode_equations(args.n)
                args.func(args, additional_arguments)
            else:
                args.func(args)
        except Exception as e:
            print(e)