import numpy as np
from typing import Callable


def modified_newton_problem(f: Callable[[float], float], f1: Callable[[float], float], f2: Callable[[float], float], p_0: float, n: int, tol=1e-5) -> None:
    """Output homework problems using newton method

    ### Parameters
    1. f : Callable[[float], float]
        - function to find the root of
    2. f1 : Callable[[float], float]
        - first derivative of the function
    3. f2 : Callable[[float], float]
        - second derivative of the function
    4. p_0 : float
        - initial guess of root
    5. n : int
        - Max number of iterations to run
    6. tol: float
        - Minimum tolerance required to find root
    """
    res = modified_newton(f, f1, f2, p_0, n=n, tol=tol, verbose=True)
    print(f'Modified Newton\'s method approximates the root p = {res[0]:.16f} after {res[1]} iterations.')


def modified_newton(f: Callable[[float], float], f1: Callable[[float], float], f2: Callable[[float], float], p_0: float, n=10, tol=1e-5, verbose=False):
    p = p_0
    if verbose:
        print(f'{"n":<10}{"p_n":<20}{"f(p_n)":<20}')
        print(f'-----------------------------------------------')
        print(f'{0:<10}{p:<20.15f}{f(p):<20.15f}')

    for i in range(1, n+1):
        p = p_0 - (f(p_0)*f1(p_0))/(f1(p_0)**2 - f(p_0)*f2(p_0))
        f_p = f(p)
        if verbose:
            print(f'{i:<10}{p:<20.15f}{f_p:<20.15f}')
        if abs(p - p_0) < tol:
            return [p, i, True]
        p_0 = p

    return [p, n, False]
