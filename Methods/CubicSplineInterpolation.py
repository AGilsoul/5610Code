import numpy as np
from typing import Callable, Union
import matplotlib.pyplot as plt


class CubicSpline:
    def __init__(self, coeffs: np.array, domains: np.array, tol=1e-5):
        """
        Constuctor for CubicSpline
        :param coeffs: Coefficients of the splines. Each row contains the coefficient for each spline (i.e., all a coefficients in row 1, all b coefficients in row 2, etc.)
        :param domains: Domains of the splines.
        :param tol: Tolerance to determine when a point is out of the bounds of the spline
        """
        assert len(coeffs[0]) == len(domains), "Number of sets of coefficients must equal number of domains provided!"
        self.a = np.array(coeffs[0], dtype=float)
        self.b = np.array(coeffs[1], dtype=float)
        self.c = np.array(coeffs[2], dtype=float)
        self.d = np.array(coeffs[3], dtype=float)
        self.domains = np.array(domains, dtype=float)
        self.n = len(coeffs[0])
        self.tol = tol

    def __call__(self, x: Union[int, float]):
        """
        Evaluate the Cubic Spline at a point
        :param x: Point to evaluate
        :return: Value of the Cubic Spline at x
        """
        s_x = 0
        computed = False
        # Find the domain which x belongs to
        for j in range(self.n):
            x_0 = self.domains[j][0]
            x_1 = self.domains[j][1]
            # Check if x belongs in this domain
            if x_0 - self.tol <= x <= x_1 + self.tol:
                # Calculate S_j(x)
                s_x = self.a[j] + self.b[j] * (x - x_0) + self.c[j] * (x - x_0)**2 + self.d[j] * (x - x_0)**3
                computed = True
                break
        assert computed, f'Given x value {x} not within domain of Cubic Spline.'
        return s_x

    def __str__(self):
        """
        Overloaded str operator for CubicSpline
        :return: String representation of class
        """
        string_out = 'S(x) =\n'
        for j in range(self.n):
            x_0 = self.domains[j][0]
            x_1 = self.domains[j][1]
            string_out += f'\tS_{j}(x) = {self.a[j]:.6f} + {self.b[j]:.6f}(x - {x_0}) + {self.c[j]:.6f}(x - {x_0})^2 + {self.d[j]:.6f}(x - {x_0})^3, {x_0} <= x <= {x_1}\n'
        return string_out


def coefficient_table(spline: CubicSpline):
    print(f'{"j":<5}{"a_j":<10}{"b_j":<10}{"c_j":<10}{"d_j":<10}')
    print(f'------------------------------')
    for j in range(spline.n):
        print(f'{j:<5}{spline.a[j]:<10.6f}{spline.b[j]:<10.6f}{spline.c[j]:<10.6f}{spline.d[j]:<10.6f}')


def spline_problem(problem_num: int, x_data: Union[list, np.array], y_data=None, f=None, boundary=None, dx=0.01):
    assert (y_data is not None or f is not None), 'Must provide either a function or y values.'
    if y_data is None:
        y_data = np.array([f(x) for x in x_data])
    n = len(x_data)
    if boundary is None:
        spline = natural_spline(x_data, y_data)
    else:
        assert (isinstance(boundary, type([])) or isinstance(boundary, type(np.array([])))), 'Boundary conditions must be either a list, tuple, or numpy array.'
        spline = clamped_spline(x_data, y_data, boundary)
    coefficient_table(spline)
    print(spline)

    x_spline = np.arange(x_data[0], x_data[n-1], dx)
    y_spline = np.array([spline(x) for x in x_spline])

    plt.plot(x_spline, y_spline, label='Cubic Spline Approximation')

    if f is not None:
        y_func = np.array([f(x) for x in x_spline])
        plt.plot(x_spline, y_func, label='Exact Solution', c='Black')

    plt.scatter(x_data, y_data, label='Data', c='Red')


    plt.title(f'Problem {problem_num}')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.show()


def multi_spline_problem(problem_num: int, x_data_set: Union[list, np.array], y_data_set: Union[list, np.array], boundary_set=Union[list, np.array, tuple], dx=0.01):
    for i in range(len(x_data_set)):
        x_data = x_data_set[i]
        y_data = y_data_set[i]
        n = len(x_data)
        spline = clamped_spline(x_data, y_data, boundary_set[i])
        print(f'Spline {i}:')
        coefficient_table(spline)

        x_spline = np.arange(x_data[0], x_data[n-1], dx)
        y_spline = np.array([spline(x) for x in x_spline])
        plt.plot(x_spline, y_spline)
        plt.scatter(x_data, y_data)

    plt.title(f'Problem {problem_num}')
    plt.xlim(0, 30)
    plt.ylim(0, 8)
    plt.xticks(np.arange(0, 30, 1))
    plt.yticks(np.arange(0, 8, 1))
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.grid()
    plt.show()


def natural_spline(x_data: Union[list, np.array], y_data: Union[list, np.array]):
    """
    Construct a cubic spline given a set of points
    :param x_data: x-coordinates of points
    :param y_data: y-coordinates of points
    :return: CubicSpline object
    """
    # Make sure data is consistent in length
    assert len(x_data) == len(y_data), "Data arrays not same length"

    # Sort input data
    sorted_list = np.array([[a, b] for a, b in sorted(zip(x_data, y_data))]).T

    # Ensure data is of float type
    x = np.array(sorted_list[0], dtype=float)
    y = np.array(sorted_list[1], dtype=float)
    n = len(x) - 1
    # Generate h and a values
    h_j = [x[j+1] - x[j] for j in range(n)]
    a_j = [y[j] for j in range(n+1)]

    c_system = []
    right_side = []
    # Construct matrix to solve for c values
    for j in range(n+1):
        c_j_row = np.zeros(n+1, dtype=float)
        # Edge cases
        if j == 0:
            c_j_row[0] = 1.0
            right_side.append(0)
        elif j == n:
            c_j_row[n] = 1.0
            right_side.append(0)
        # Regular cases
        else:
            c_j_row[j-1] = h_j[j-1]
            c_j_row[j] = 2.0 * (h_j[j] + h_j[j-1])
            c_j_row[j+1] = h_j[j]
            right_side.append((3.0/h_j[j])*(a_j[j+1] - a_j[j]) - (3.0/h_j[j-1])*(a_j[j] - a_j[j-1]))
        c_system.append(c_j_row)

    # Solve for c values
    c_system = np.array(c_system, dtype=float)
    c_j = np.linalg.solve(c_system, right_side)
    # compute b and d values
    b_j = [(a_j[j+1] - a_j[j])/h_j[j] - (1.0/3.0)*(c_j[j+1] + 2*c_j[j])*h_j[j] for j in range(n)]
    d_j = [(c_j[j+1] - c_j[j])/(3*h_j[j]) for j in range(n)]

    # Get final coefficients and domains for each spline
    final_coeffs = np.array([a_j[0:n], b_j[0:n], c_j[0:n], d_j[0:n]])
    domains = [[x[j], x[j+1]] for j in range(n)]

    # Return the CubicSpline object
    return CubicSpline(final_coeffs, domains)


def clamped_spline(x_data: Union[list, np.array], y_data: Union[list, np.array], boundary: Union[list, np.array]):
    """
    Construct a cubic spline given a set of points
    :param x_data: x-coordinates of points
    :param y_data: y-coordinates of points
    :return: CubicSpline object
    """
    # Make sure data is consistent in length
    assert len(x_data) == len(y_data), "Data arrays not same length"

    # Sort input data
    sorted_list = np.array([[a, b] for a, b in sorted(zip(x_data, y_data))]).T

    # Ensure data is of float type
    x = np.array(sorted_list[0], dtype=float)
    y = np.array(sorted_list[1], dtype=float)
    n = len(x) - 1
    # Generate h and a values
    h_j = [x[j+1] - x[j] for j in range(n)]
    a_j = [y[j] for j in range(n+1)]

    # Weird algorithm for tridiagonal systems!
    alpha = np.zeros(n+1, dtype=float)
    alpha[0] = 3 * (a_j[1] - a_j[0]) / h_j[0] - 3 * boundary[0]
    alpha[n] = 3 * boundary[1] - 3 * (a_j[n] - a_j[n-1]) / h_j[n-1]

    for j in range(1, n):
        alpha[j] = (3 / h_j[j]) * (a_j[j+1] - a_j[j]) - (3 / h_j[j-1]) * (a_j[j] - a_j[j-1])

    l = np.zeros(n+1, dtype=float)
    mu = np.zeros(n+1, dtype=float)
    z = np.zeros(n+1, dtype=float)
    c_j = np.zeros(n+1, dtype=float)
    b_j = np.zeros(n+1, dtype=float)
    d_j = np.zeros(n+1, dtype=float)

    l[0] = 2 * h_j[0]
    mu[0] = 0.5
    z[0] = alpha[0] / l[0]

    for j in range(1, n):
        l[j] = 2 * (x_data[j+1] - x_data[j-1]) - h_j[j-1] * mu[j-1]
        mu[j] = h_j[j] / l[j]
        z[j] = (alpha[j] - h_j[j-1] * z[j-1]) / l[j]

    l[n] = h_j[n-1] * (2 - mu[n-1])
    z[n] = (alpha[n] - h_j[n-1] * z[n-1]) / l[n]
    c_j[n] = z[n]

    for j in reversed(range(n)):
        c_j[j] = z[j] - mu[j] * c_j[j+1]
        b_j[j] = (a_j[j+1] - a_j[j]) / h_j[j] - h_j[j] * (c_j[j+1] + 2 * c_j[j]) / 3
        d_j[j] = (c_j[j+1] - c_j[j]) / (3 * h_j[j])

    # No clue why this doesn't work?
    '''
    c_system = []
    alpha = np.zeros(n+1, dtype=float)
    # Construct matrix to solve for c values
    for j in range(n+1):
        c_j_row = np.zeros(n+1, dtype=float)
        # Edge cases
        if j == 0:
            c_j_row[0] = 2.0 * h_j[0]
            c_j_row[1] = h_j[0]
            alpha[0] = (3.0 * (a_j[1] - a_j[0]) / h_j[0]) - (3.0 * boundary[0])
        elif j == n:
            c_j_row[n] = h_j[n-1]
            c_j_row[n] = 2.0 * h_j[n-1]
            alpha[n] = (3.0 * boundary[1]) - (3.0 * (a_j[n] - a_j[n-1]) / h_j[n-1])
        # Regular cases
        else:
            c_j_row[j-1] = h_j[j-1]
            c_j_row[j] = 2.0 * (h_j[j] + h_j[j-1])
            c_j_row[j+1] = h_j[j]
            alpha[j] = (((3.0/h_j[j]) * (a_j[j+1] - a_j[j])) - ((3.0/h_j[j-1]) * (a_j[j] - a_j[j-1])))
        c_system.append(c_j_row)

    ###
    ###
    ###

    # Solve for c values
    c_system = np.array(c_system, dtype=float)
    c_j = np.linalg.solve(c_system, alpha)
    # compute b and d values
    b_j = [(a_j[j+1] - a_j[j])/h_j[j] - (h_j[j]/3.0)*(c_j[j+1] + 2*c_j[j]) for j in range(n)]
    d_j = [(c_j[j+1] - c_j[j])/(3*h_j[j]) for j in range(n)]
    '''

    # Get final coefficients and domains for each spline
    final_coeffs = np.array([a_j[0:n], b_j[0:n], c_j[0:n], d_j[0:n]])
    domains = [[x[j], x[j+1]] for j in range(n)]

    # Return the CubicSpline object
    return CubicSpline(final_coeffs, domains)
