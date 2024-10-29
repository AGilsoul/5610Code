import numpy as np
from typing import Callable


def newton_problem(f: Callable[[float], float], df_dx: Callable[[float], float], p_0: float, n: int, tol=1e-5) -> None:
    """Output homework problems using newton method

    ### Parameters
    1. f : Callable[[float], float]
        - function to find the root of
    2. df_dx : Callable[[float,], float]
        - derivative of the function
    3. p_0 : float
        - initial guess of root
    4. n : int
        - Max number of iterations to run
    5. tol: float
        - Minimum tolerance required to find root
    """
    res = newton(f, df_dx, p_0, n=n, tol=tol, verbose=True)
    print(f'Newton\'s method approximated the root p = {res[0]:.6f} after {res[1]} iterations.')


def secant_problem(f: Callable[[float], float], p_0: float, p_1: float, n: int, tol=1e-5) -> None:
    """Output homework problems using secant method

    ### Parameters
    1. f : Callable[[float], float]
        - function to find the root of
    2. p_0 : float
        - initial guess 1
    3. p_1 : float
        - initial guess 2
    4. n : int
        - Max number of iterations to run
    5. tol: float
        - Minimum tolerance required to find root
    """
    res = secant(f, p_0, p_1, n=n, tol=tol, verbose=True)
    print(f'Secant method approximated the root p = {res[0]:.6f} after {res[1]} iterations.')


def false_position_problem(f: Callable[[float], float], p_0: float, p_1: float, n: int, tol=1e-5) -> None:
    """Output homework problems using method of false position

    ### Parameters
    1. f : Callable[[float], float]
        - function to find the root of
    2. p_0 : float
        - initial guess 1
    3. p_1 : float
        - initial guess 2
    4. n : int
        - Max number of iterations to run
    5. tol: float
        - Minimum tolerance required to find root
    """
    res = false_position(f, p_0, p_1, n=n, tol=tol, verbose=True)
    print(f'Method of False Position approximated the root p = {res[0]:.6f} after {res[1]} iterations.')


def newton(f: Callable[[float], float], df_dx: Callable[[float], float], p_0: float, n=10, tol=1e-5, verbose=False):
    p = p_0
    if verbose:
        print(f'{"n":<10}{"p_n":<10}{"f(p_n)":<10}')
        print(f'------------------------------')
        print(f'{0:<10}{p:<10.6f}{f(p):<10.6f}')
    for i in range(1, n+1):
        p = p_0 - f(p_0) / df_dx(p_0)
        f_p = f(p)
        if verbose:
            print(f'{i:<10}{p:<10.6f}{f_p:<10.6f}')
        if abs(p - p_0) < tol:
            return [p, i, True]
        p_0 = p

    return [p, n, False]


def secant(f: Callable[[float], float], p_0: float, p_1: float, n=10, tol=1e-5, verbose=False):
    p = 0
    if verbose:
        print(f'{"n":<10}{"p_n":<10}{"f(p_n)":<10}')
        print(f'------------------------------')
        print(f'{0:<10}{p_0:<10.6f}{f(p_0):<10.6f}')
        print(f'{1:<10}{p_1:<10.6f}{f(p_1):<10.6f}')
    for i in range(2, n + 1):
        p = p_1 - ((p_0 - p_1)*f(p_1)) / (f(p_0) - f(p_1))
        f_p = f(p)
        if verbose:
            print(f'{i:<10}{p:<10.6f}{f_p:<10.6f}')
        if abs(p - p_1) < tol:
            return [p, i, True]
        p_0 = p_1
        p_1 = p

    return [p, n, False]


def false_position(f: Callable[[float], float], p_0: float, p_1: float, n=10, tol=1e-5, verbose=False):
    q_0 = f(p_0)
    q_1 = f(p_1)
    q = 0
    p = 0
    if q_0 == 0:
        return p_0
    elif q_1 == 0:
        return p_1
    elif q_0 * q_1 > 0:
        raise ValueError('Invalid interval')
    if verbose:
        print(f'{"n":<10}{"p_n":<10}{"f(p_n)":<10}')
        print(f'------------------------------')
        print(f'{0:<10}{p_0:<10.6f}{f(p_0):<10.6f}')
        print(f'{1:<10}{p_1:<10.6f}{f(p_1):<10.6f}')

    for i in range(2, n+1):
        p = p_1 - q_1*(p_1 - p_0)/(q_1 - q_0)
        q = f(p)
        if verbose:
            print(f'{i:<10}{p:<10.6f}{q:<10.6f}')
        if abs(p - p_1) < tol:
            return [p, i, True]
        if q * q_1 > 0:
            p_0 = p_1
            q_0 = q_1
        else:
            p_1 = p
            q_1 = q

    return [p, n, False]





