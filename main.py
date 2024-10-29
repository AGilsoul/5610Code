from Methods.Bisection import bisection
from Methods.FixedPoint import fixed_point
import numpy as np


def bisection_test_fcn(x: float) -> float:
    return np.sqrt(x) - np.cos(x)


def fixed_point_test_fcn(x: float) -> float:
    return x - (x**3 + 4 * x**2 - 10) / (3 * x**2 + 8 * x)


def main():
    print(f'Bisection')
    res = bisection(bisection_test_fcn, (0, 2), n=100)
    if res[2]:
        print(f'Successful after {res[1]} iterations')
        print(f'x={res[0]}, f({res[0]}) = {bisection_test_fcn(res[0])}')
    else:
        print(f'Unsuccessful after {res[1]} iterations')
    print(f'\nFixed Point Iteration')
    res = fixed_point(fixed_point_test_fcn, 1)
    if res[2]:
        print(f'Successful after {res[1]} iterations')
        print(f'x={res[0]}, f({res[0]}) = {fixed_point_test_fcn(res[0])}')
    else:
        print(f'Unsuccessful after {res[1]} iterations')



if __name__ == '__main__':
    main()
