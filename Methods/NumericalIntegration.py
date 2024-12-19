import numpy as np
from typing import Callable


def composite_trapezoidal_intg(f: Callable[[float], float], a: float, b: float, n: int):
    h = (b - a) / n
    res = f(a) + f(b)
    for j in range(1, n):
        x_j = a + j * h
        res += 2.0 * f(x_j)

    return res * (h / 2.0)


def composite_simpson_intg(f: Callable[[float], float], a: float, b: float, n: int):
    h = (b - a) / n
    assert n % 2 == 0, 'n must be even!'
    res = f(a) + f(b) + 4 * f(a + (n-1) * h)
    for j in range(1, int(n/2)):
        x_index = a + 2 * j * h
        res += 2 * f(x_index) + 4 * f(x_index + h)

    return res * (h / 3)

