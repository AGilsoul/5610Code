from Methods.Bisection import bisection_problem
from Methods.FixedPoint import fixed_point_problem
import numpy as np


def problem5():
    def f(x: float) -> float:
        return x ** 3 + 4 * x ** 2 - 10
    print(f'_____________________________________')
    print(f'Problem 5')
    bisection_problem(f, (1, 2), n=100, tol=1e-4)
    print()


def problem6():
    def f(x: float) -> float:
        return 3*x - np.exp(x)
    print(f'_____________________________________')
    print(f'Problem 6')
    bisection_problem(f, (1, 2), n=100, tol=1e-5)
    print()


def problem7():
    def f(x: float) -> float:
        return x**3 - x - 1
    print(f'_____________________________________')
    print('Problem 7')
    bisection_problem(f, (1, 2), n=100, tol=1e-4)
    print()


def problem10():
    def f(x: float) -> float:
        return x - ((x**3 + 4 * x**2 - 10) / (3 * x**2 + 8 * x))
    print(f'_____________________________________')
    print('Problem 10')
    fixed_point_problem(f, 1.5, n=100, tol=1e-9)
    print()


def problem11():
    def g1(x: float) -> float:
        return (3 + x - 2 * x**2)**(1/4)

    def g2(x: float) -> float:
        return ((x + 3 - x**4)/2)**(1/2)

    def g3(x: float) -> float:
        return ((x+3)/(x+2))**(1/2)

    def g4(x: float) -> float:
        return (3 * x**4 + 2 * x**2 + 3)/(4 * x**3 + 4 * x - 1)

    print(f'_____________________________________')
    print(f'Problem 11')
    print(f'g1(x)')
    fixed_point_problem(g1, 1, n=4, tol=1e-9)
    print(f'g2(x)')
    fixed_point_problem(g2, 1, n=4, tol=1e-9)
    print(f'g3(x)')
    fixed_point_problem(g3, 1, n=4, tol=1e-9)
    print(f'g4(x)')
    fixed_point_problem(g4, 1, n=4, tol=1e-9)
    print()


def problem12():
    def g(x: float) -> float:
        return np.pi + 0.5 * np.sin(x / 2)
    print(f'_____________________________________')
    print(f'Problem 12')
    fixed_point_problem(g, np.pi, n=100, tol=1e-2)
    print()


def main():
    problem5()
    problem6()
    problem7()
    problem10()
    problem11()
    problem12()


if __name__ == '__main__':
    main()


