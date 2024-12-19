import numpy as np
from copy import deepcopy
from abc import ABC, abstractmethod


class TrialFunction(ABC):
    @abstractmethod
    def __add__(self, other):
        """
        Returns a new function as the sum of this function and another

        :param other: Function to be added
        :return: Sum function
        """
        pass

    @abstractmethod
    def __iadd__(self, other):
        """
        Returns this function when another function is added to it

        :param other: Function to be added
        :return: This function as the sum
        """
        pass

    @abstractmethod
    def __call__(self, x: float):
        """
        Returns the value of this function at x

        :param x: Value to evaluate function at
        :return: Value of function at x
        """
        pass

    @abstractmethod
    def __str__(self):
        """
        Displays the function as a string

        :return: Output string
        """
        pass

    @abstractmethod
    def integrate(self, a: float, b: float):
        """
        Returns the area under this function bounded from x=a to x=b

        :param a: Left bound
        :param b: Right bound
        :return: Resultant integral
        """
        pass

    @abstractmethod
    def differentiate(self):
        """
        Returns the function representing the derivative of this function

        :return: Derivative function
        """
        pass


class Polynomial(TrialFunction):
    def __init__(self, coeffs):
        self.c = coeffs
        self.n = len(coeffs) + 1
        return

    def __add__(self, other):
        if self.n < other.n:
            coeffs = [self.c[i] + other.c[i] for i in range(other.n - 3)]
            coeffs.extend(other.c[other.n - 3:])
        else:
            coeffs = [self.c[i] + other.c[i] for i in range(self.n - 3)]
            coeffs.extend(self.c[self.n - 3:])
        return Polynomial(coeffs)

    def __iadd__(self, other):
        for i in range(min(self.n, other.n) - 1):
            self.c[i] += other.c[i]
        if self.n < other.n:
            self.c.extend(other.c[other.n - 3:])
        return self

    def __call__(self, x: float):
        return sum(self.c[i]*x**i for i in range(self.n - 1))

    def __str__(self):
        out_str = f'{self.c[0]}'
        for i in range(1, self.n - 1):
            out_str += f' + {self.c[i]}*x^{i}'
        return out_str

    def integrate(self, a: float, b: float):
        return sum((self.c[i]/(i+1))*b**(i+1) - (self.c[i]/(i+1))*a**(i+1) for i in range(self.n - 1))

    def differentiate(self):
        return Polynomial([self.c[i] * i for i in range(1, self.n - 1)])


def weighted_residual_method(f: TrialFunction):
    return



poly = Polynomial([1, 0, 1])
print(poly)
print(poly.integrate(0, 1))
print(poly.differentiate())

help(Polynomial)
