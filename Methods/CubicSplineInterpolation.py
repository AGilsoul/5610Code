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
            string_out += f'\tS_{j}(x) = {self.a[j]} + {self.b[j]}(x - {x_0}) + {self.c[j]}(x - {x_0})^2 + {self.d[j]}(x - {x_0})^3, {x_0} <= x <= {x_1}\n'
        return string_out


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
    print(x)
    print(domains)

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

    c_system = []
    right_side = []
    # Construct matrix to solve for c values
    for j in range(n+1):
        c_j_row = np.zeros(n+1, dtype=float)
        # Edge cases
        if j == 0:
            c_j_row[0] = 2.0 * h_j[0]
            c_j_row[1] = h_j[0]
            right_side.append((3.0/h_j[j]) * (a_j[j+1] - a_j[j]) - boundary[0])
        elif j == n:
            c_j_row[n] = h_j[j-1]
            c_j_row[n] = 2.0 * h_j[j-1]
            right_side.append(3.0 * boundary[1] - (3.0 / h_j[j-1]) * (a_j[j] - a_j[j-1]))
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
    print(x)
    print(domains)

    # Return the CubicSpline object
    return CubicSpline(final_coeffs, domains)



num_points = 20
X = np.linspace(0, 1, num_points)
Y = abs(np.random.normal(size=num_points))
Y = Y / max(Y)
print(Y)
X[0] = 0
X[19] = 1
Y[0] = 0.5
Y[1] = 0.5

bc = [0.0, 0.0]

x_min = min(X)
x_max = max(X)
dx = 0.0001
# res = natural_spline(X, Y)
res = clamped_spline(X, Y, bc)
print(res)


x_eval = np.arange(x_min, x_max+dx, dx)
y_eval = []
for x in x_eval:
    y_eval.append(res(x))

plt.plot(x_eval, y_eval)
plt.xlim(-0.1, 1.1)
plt.ylim(-0.1, 1.1)
plt.scatter(X, Y)
plt.show()
