from Methods.NumericalIntegration import *
import numpy as np


def problem6():
    def f(x: float) -> float:
        return np.sin(x)
    print(f'_____________________________________')
    print(f'Problem 6')
    n = 360
    a = 0.0
    b = np.pi
    result = composite_trapezoidal_intg(f, a, b, n)
    print(f'The approximate value of the integral using Composite Trapezoidal Rule with n = {n} is {result}')
    return


def problem7():
    def f(x: float) -> float:
        return np.sin(x)
    print(f'_____________________________________')
    print(f'Problem 7')
    n = 18
    a = 0.0
    b = np.pi
    result = composite_simpson_intg(f, a, b, n)
    print(f'The approximate value of the integral using Composite Simpson\'s Rule with n = {n} is {result}')
    return


def problem8():
    def f(x: float) -> float:
        return x**2 * np.log(x**2 + 1)
    print(f'_____________________________________')
    print(f'Problem 8')
    n_trap = 8
    n_simp = 8
    a = 0
    b = 2
    res_trap = composite_trapezoidal_intg(f, a, b, n_trap)
    res_simp = composite_simpson_intg(f, a, b, n_simp)
    print(f'The approximate value of the integral using Composite Trapezoidal Rule with n = {n_trap} is {res_trap}')
    print(f'The approximate value of the integral using Composite Simpson\'s Rule with n = {n_simp} is {res_simp}')


def problem9():
    def f(x: float) -> float:
        return 1.0 / (x + 4.0)
    print(f'_____________________________________')
    print(f'Problem 9')
    n_trap = 46
    n_simp = 6
    a = 0
    b = 2
    res_trap = composite_trapezoidal_intg(f, a, b, n_trap)
    res_simp = composite_simpson_intg(f, a, b, n_simp)
    print(f'The approximate value of the integral using Composite Trapezoidal Rule with n = {n_trap} is {res_trap}')
    print(f'The approximate value of the integral using Composite Simpson\'s Rule with n = {n_simp} is {res_simp}')


def main():
    problem6()
    problem7()
    problem8()
    problem9()


if __name__ == '__main__':
    main()