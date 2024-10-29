from Methods.ModifiedNewtonMethod import modified_newton_problem
from Methods.NewtonMethod import newton_problem
import numpy as np


def problem2():
    def f(x: float) -> float:
        return np.exp(x) - x - 1

    def f1(x: float) -> float:
        return np.exp(x) - 1

    def f2(x: float) -> float:
        return np.exp(x)

    p0 = 1
    tol = 1e-10
    n = 100
    print('Problem 2')
    modified_newton_problem(f, f1, f2, p0, n=n, tol=tol)
    print()


def problem3a():
    def f(x: float) -> float:
        return x**2 - 2*x*np.exp(-x) + np.exp(-2*x)

    def f1(x: float) -> float:
        return 2*x - 2*(np.exp(-x) - x*np.exp(-x)) - 2*np.exp(-2*x)

    p0 = 0.5
    tol = 1e-4
    n = 100
    print('Problem 3a')
    newton_problem(f, f1, p0, n=n, tol=tol)
    print()


def problem3b():
    def f(x: float) -> float:
        return x**2 - 2*x*np.exp(-x) + np.exp(-2*x)

    def f1(x: float) -> float:
        return 2*x - 2*(np.exp(-x) - x*np.exp(-x)) - 2*np.exp(-2*x)

    def f2(x: float) -> float:
        return 2 - 2*(x*np.exp(-x) - 2*np.exp(-x)) + 4*np.exp(-2*x)

    p0 = 0.5
    tol = 1e-4
    n = 100
    print('Problem 3b')
    modified_newton_problem(f, f1, f2, p0, n=n, tol=tol)
    print()


def main():
    problem2()
    problem3a()
    problem3b()
    return


if __name__ == '__main__':
    main()