"""
Provide some helpful functions for DE
"""

from typing import Tuple


def extract_iv(eq: str) -> Tuple[float, float]:
    """
    Extract initial value for Cauchy problem

    Parameters
    ----------
    eq: str
        Condition
    Returns
    -------
    iv: tuple[float, float]
        Pair of point and value at point
    """
    left_v = float(eq[eq.index("(") + 1:eq.index(")")])
    right_v = float(eq[eq.index("=") + 1:])
    return left_v, right_v
