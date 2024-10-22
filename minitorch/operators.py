"""Collection of the core mathematical operators used throughout the code base."""

import math
from typing import Callable, Iterable

# ## Task 0.1


def mul(x: float, y: float) -> float:
    """Multplies two arguments"""
    return x * y


def id(a: float) -> float:
    """Returns the argument unchanged"""
    return a


def add(x: float, y: float) -> float:
    """Adds two arguments"""
    return x + y


def neg(x: float) -> float:
    """Negates the argument"""
    return -x


def lt(x: float, y: float) -> float:
    """Checks if x is less than y

    Args:
    ----
      x (float): The first input value.
      y (float): The second input value.

    Returns:
    -------
      float: 1.0 if x < y, 0.0 otherwise.

    """
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Checks if two values x and y are equal"""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Returns the maximum out of two input values"""
    return x if x > y else y


def is_close(x: float, y: float) -> float:
    """Checks if input values are close"""
    return (x - y < 1e-2) and (y - x < 1e-2)


def sigmoid(x: float) -> float:
    """Computes sigmoid of given input x"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        return math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Applies the Rectified Linear Unit (ReLU) activation function to the input x"""
    return x if x > 0 else 0.0


EPS = 1e-6


def log(x: float) -> float:
    """Compute the natural logarithm of a given input"""
    return math.log(x + EPS)


def exp(x: float) -> float:
    """Returns the exponential function of input value"""
    return math.exp(x)


def inv(x: float) -> float:
    """Returns the inverse of input value"""
    return 1.0 / x


def log_back(x: float, y: float) -> float:
    """Computes the derivative of log times a second arg"""
    return y / (x + EPS)


def inv_back(x: float, y: float) -> float:
    """Computes the derivative of reciprocal times a second arg"""
    return (-1.0 / (x**2)) * y


def relu_back(x: float, y: float) -> float:
    """Computes the derivative of ReLU of x times a second arg y."""
    return y if x > 0 else 0.0


def sigmoid_back(x: float, y: float) -> float:
    """Computes the derivative of sigmoid of x times a second arg y."""
    return y * (1 - y)


def exp_back(x: float, y: float) -> float:
    """Computes the derivative of exponential of x times a second arg y."""
    return y * math.exp(x)


# ## Task 0.3


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Map a function to a list"""

    def _map(ls: Iterable[float]) -> Iterable[float]:
        ret = []
        for x in ls:
            ret.append(fn(x))
        return ret

    return _map


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Higher-order function that combines elements from two iterables using a given function

    Args:
    ----
      fn (Callable): The function used to combine elements from list1 and list2
      list1 (Iterable): The first iterable containing elements to be combined
      list2 (Iterable): The second iterable containing elements to be combined

    Returns:
    -------
      list: A new list containing the combined elements

    """

    def _zipWith(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
        ret = []
        for x, y in zip(ls1, ls2):
            ret.append(fn(x, y))
        return ret

    return _zipWith


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    r"""Higher-order reduce.

    Args:
    ----
        fn: combine two values
        start: start value $x_0$

    Returns:
    -------
        Function that takes a list `ls` of elements
        $x_1 \ldots x_n$ and computes the reduction :math:`fn(x_3, fn(x_2,
        fn(x_1, x_0)))`

    """

    # ASSIGN0.3
    def _reduce(ls: Iterable[float]) -> float:
        val = start
        for l in ls:
            val = fn(val, l)
        return val

    return _reduce


def negList(list1: Iterable[float]) -> Iterable[float]:
    """Negate all elements in a list using map"""
    return map(neg)(list1)


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    """Adds corresponding elements from two iterables using zipWith

    Args:
    ----
        ls1: The first iterable of floats
        ls2: The second iterable of floats

    Returns:
    -------
        An iterable containing the sum of corresponding elements

    """
    return zipWith(add)(ls1, ls2)


def sum(list1: Iterable[float]) -> float:
    """Returns sum of all elements in a list using reduce"""
    return reduce(add, 0.0)(list1)


def prod(list1: Iterable[float]) -> float:
    """Calculates the product of all elements in a list using reduce"""
    return reduce(mul, 1.0)(list1)
