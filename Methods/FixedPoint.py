import numpy as np
from typing import Callable


def fixed_point_problem(f: Callable[[float], float], p_0: float, n: int, tol=1e-5) -> None:
    """Output homework problems using fixed point iteration

    ### Parameters
    1. f : Callable[[float], float]
        - function to find the root of
    2. p_0 : float
        - initial guess of fixed point
    3. n : int
        - Max number of iterations to run
    4. tol: float
        - Minimum tolerance required to find root
    """
    res = fixed_point(f, p_0, n=n, tol=tol, verbose=True)
    print(f'Fixed-Point Iteration approximated the fixed-point p = {res[0]:.6f} after {res[1]} iterations.')


def fixed_point(fcn: Callable[[float], float], p_0: float, n=10, tol=1e-5, verbose=False) -> list:
    """Implements fixed point iteration method

    ### Parameters
    1. fcn : Callable[[float], float]
        - function to find the fixed point of
    2. p_0 : float
        - initial guess of fixed point
    3. n : int
        - Max number of iterations to run
    4. tol: float
        - Minimum tolerance required to find root
    5. verbose: bool
        - Verbosity level

    ### Returns
    - List
        - List containing the approximate fixed point, number of iterations done, and True if method was successful
    """
    if verbose:
        print(f'{"n":<10}{"p_n":<10}{"g(p_n)":<10}')
        print(f'------------------------------')
    for i in range(n+1):
        p = fcn(p_0)
        if verbose:
            print(f'{i:<10}{p_0:<10.6f}{p:<10.6f}')
        if abs(p - p_0) < tol:
            return [p, i, True]
        p_0 = p

    return [p_0, n, False]
