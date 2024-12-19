from Methods.CubicSplineInterpolation import spline_problem, multi_spline_problem
import numpy as np


def problem1():
    print(f'_____________________________________')
    print(f'Problem 1')

    def f(x: float) -> float:
        return np.exp(x)
    x_data = np.array([0, 1, 2, 3])
    spline_problem(1, x_data, f=f)
    print()
    return


def problem2():
    print(f'_____________________________________')
    print(f'Problem 2')
    x_data = np.array([0.1, 0.2, 0.3, 0.4])
    y_data = np.array([-0.62049958, -0.28398668, 0.00660095, 0.24842440])
    spline_problem(2, x_data, y_data=y_data)
    print()
    return


def problem3():
    print(f'_____________________________________')
    print(f'Problem 3')

    def f(x: float) -> float:
        return np.exp(x)
    x_data = np.array([0, 1, 2, 3])
    bc = np.array([1, np.exp(3)])
    spline_problem(3, x_data, f=f, boundary=bc)
    print()
    return


def problem4():
    print(f'_____________________________________')
    print(f'Problem 4')
    x_data = np.array([0.1, 0.2, 0.3, 0.4])
    y_data = np.array([-0.62049958, -0.28398668, 0.00660095, 0.24842440])
    bc = np.array([3.58502082, 2.16529366])
    spline_problem(4, x_data, y_data=y_data, boundary=bc)
    print()
    return


def problem5():
    print(f'_____________________________________')
    print(f'Problem 5')
    x_data_set = [[1, 2, 5, 6, 7, 8, 10, 13, 17],
                  [17, 20, 23, 24, 25, 27, 27.7],
                  [27.7, 28, 29, 30]]
    y_data_set = [[3.0, 3.7, 3.9, 4.2, 5.7, 6.6, 7.1, 6.7, 4.5],
                  [4.5, 7.0, 6.1, 5.6, 5.8, 5.2, 4.1],
                  [4.1, 4.3, 4.1, 3.0]]
    bc_set = [[1.0, -0.67],
              [3.0, -4.0],
              [0.33, -1.5]]
    multi_spline_problem(5, x_data_set, y_data_set, bc_set, dx=0.001)
    print()


def main():
    problem1()
    problem2()
    problem3()
    problem4()
    problem5()
    return


if __name__ == '__main__':
    main()