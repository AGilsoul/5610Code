from typing import Callable, Union


def bisection_problem(f: Callable[[float], float], interval: tuple, n: int, tol=1e-5) -> None:
    """Output homework problems using the Bisection method

    ### Parameters
    1. f : Callable[[float], float]
        - function to find the root of
    2. interval : tuple
        - interval to search for root in
    3. n : int
        - Max number of iterations to run
    4. tol: float
        - Minimum tolerance required to find root
    """
    res = bisection(f, interval, n=n, tol=tol, verbose=True)
    print(f'The Bisection Method approximates the root p = {res[0]:.6f} after {res[1]} iterations.')



def bisection(fcn: Callable[[float], float], interval: tuple, n: int, tol=1e-5, verbose=False) -> list:
    """Implements bisection root finding method

    ### Parameters
    1. fcn : Callable[[float], float]
        - function to find the root of
    2. interval : tuple
        - interval to search for root in
    3. n : int
        - Max number of iterations to run
    4. tol: float
        - Minimum tolerance required to find root
    5. verbose: bool
        - Verbosity level

    ### Returns
    - List
        - List containing the approximate root, number of iterations done, and True if method was successful

    Raises
    ------
    - ValueError
        - Thrown if the the interval is invalid
    """
    a, b = interval
    f_a, f_b = fcn(a), fcn(b)
    if f_a == 0:
        return a
    elif f_b == 0:
        return b
    elif f_a * f_b > 0:
        raise ValueError('Invalid interval')

    p = 0
    if verbose:
        print(f'{"n":<10}{"p_n":<10}{"f(p_n)":<10}')
        print(f'------------------------------')
    for i in range(1, n+1):
        p = (a + b) / 2
        f_p = fcn(p)
        if verbose:
            print(f'{i:<10}{p:<10.6f}{f_p:<10.6f}')
        if abs(f_p - 0) < tol:
            return [p, i, True]
        elif f_p * fcn(a) < 0:
            b = p
        else:
            a = p

    return [p, n, True]

