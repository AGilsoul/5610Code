from Methods.NewtonMethod import *
import numpy as np


def problem4():
    def f(x: float) -> float:
        return np.cos(x) - x

    def df_dx(x: float) -> float:
        return -np.sin(x) - 1

    p0 = np.pi / 4
    tol = 1e-9
    n = 100
    print('Problem 4')
    newton_problem(f, df_dx, p0, n, tol)
    print()


def problem5():
    def f(x: float) -> float:
        return np.cos(x) - x

    p0 = 0.5
    p1 = np.pi / 4
    tol = 1e-9
    n = 100
    print('Problem 5')
    secant_problem(f, p0, p1, n, tol)
    print()


def problem6():
    def f(x: float) -> float:
        return np.cos(x) - x

    p0 = 0.5
    p1 = np.pi / 4
    tol = 1e-9
    n = 100
    print('Problem 6')
    false_position_problem(f, p0, p1, n, tol)
    print()


def problem7a():
    def f(x: float) -> float:
        return x**3 - 2*x**2 - 5

    def df_dx(x: float) -> float:
        return 3*x**2 - 4*x

    p0 = 2.5
    tol = 1e-4
    n = 100
    print('Problem 7a')
    newton_problem(f,  df_dx, p0, n, tol)
    print()


def problem7b():
    def f(x: float) -> float:
        return x ** 3 - 2 * x ** 2 - 5

    p0 = 2.5
    p1 = 3
    tol = 1e-4
    n = 100
    print('Problem 7b')
    secant_problem(f, p0, p1, n, tol)
    print()


def problem7c():
    def f(x: float) -> float:
        return x ** 3 - 2 * x ** 2 - 5

    p0 = 2.5
    p1 = 3
    tol = 1e-4
    n = 100
    print('Problem 7c')
    false_position_problem(f, p0, p1, n, tol)
    print()


def main():
    problem4()
    problem5()
    problem6()
    problem7a()
    problem7b()
    problem7c()
    return


if __name__ == '__main__':
    main()

