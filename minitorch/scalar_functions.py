from __future__ import annotations

from typing import TYPE_CHECKING

import minitorch

from . import operators
from .autodiff import Context

if TYPE_CHECKING:
    from typing import Tuple

    from .scalar import Scalar, ScalarLike


def wrap_tuple(x: float | Tuple[float, ...]) -> Tuple[float, ...]:
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


class ScalarFunction:
    """A wrapper for a mathematical function that processes and produces
    Scalar variables.

    This is a static class and is never instantiated. We use `class`
    here to group together the `forward` and `backward` code.
    """

    @classmethod
    def _backward(cls, ctx: Context, d_out: float) -> Tuple[float, ...]:
        """Compute the derivative of the function with respect to the output gradient 'd_out'."""
        return wrap_tuple(cls.backward(ctx, d_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: float) -> float:
        """Compute the forward pass of the function."""
        return float(cls.forward(ctx, *inps))  # type: ignore

    @classmethod
    def apply(cls, *vals: ScalarLike) -> Scalar:
        """Apply the function to the input values."""
        raw_vals = []
        scalars = []
        for v in vals:
            if isinstance(v, minitorch.scalar.Scalar):
                scalars.append(v)
                raw_vals.append(v.data)
            else:
                scalars.append(minitorch.scalar.Scalar(v))
                raw_vals.append(v)

        # Create the context.
        ctx = Context(False)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)
        assert isinstance(c, float), "Expected return type float got %s" % (type(c))

        # Create a new variable from the result with a new history.
        back = minitorch.scalar.ScalarHistory(cls, ctx, scalars)
        return minitorch.scalar.Scalar(c, back)


# Examples
class Add(ScalarFunction):
    """Addition function $f(x, y) = x + y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the sum of the inputs."""
        return a + b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the derivative of the addition function with respect to the output gradient 'd_output'."""
        return d_output, d_output


class Mul(ScalarFunction):
    """Multiplication function $f(x, y) = x * y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the product of the inputs."""
        ctx.save_for_backward(a, b)  # Save a and b for backward pass
        return a * b

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the derivative of the multiplication function with respect to the output gradient 'd_output'."""
        a, b = ctx.saved_tensors  # Retrieve saved values of a and b
        # The derivative of a * b with respect to a is b, and with respect to b is a
        return d_output * b, d_output * a


class Neg(ScalarFunction):
    """Negation function $f(x) = -x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the negation of the input."""
        return -a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the derivative of the negation function with respect to the output gradient 'd_output'."""
        return -d_output


class Inv(ScalarFunction):
    """Inverse function $f(x) = 1/x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the inverse of the input."""
        ctx.save_for_backward(a)
        return 1 / a

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the derivative of the inverse function with respect to the output gradient 'd_output'."""
        (a,) = ctx.saved_values
        return -d_output / a**2


class Exp(ScalarFunction):
    """Exponential function $f(x) = e^x$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the exponential of the input."""
        ctx.save_for_backward(a)
        return operators.exp(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the derivative of the exponential function with respect to the output gradient 'd_output'."""
        (a,) = ctx.saved_values
        return operators.exp_back(a, d_output)


class EQ(ScalarFunction):
    """Equality function $f(x, y) = x == y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Compute the equality of the inputs."""
        return operators.eq(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the derivative of the equality function with respect to the output gradient 'd_output'."""
        return 0.0, 0.0


class LT(ScalarFunction):
    """Less than function $f(x, y) = x < y$"""

    @staticmethod
    def forward(ctx: Context, a: float, b: float) -> float:
        """Return the result of the less than operation."""
        return operators.lt(a, b)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> Tuple[float, ...]:
        """Compute the derivative of the less than function with respect to the output gradient 'd_output'."""
        return 0.0, 0.0


class ReLU(ScalarFunction):
    """ReLU function $f(x) = max(0, x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the ReLU of the input."""
        ctx.save_for_backward(a)
        return operators.relu(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the derivative of the ReLU function with respect to the output gradient 'd_output'."""
        (a,) = ctx.saved_values
        return operators.relu_back(a, d_output)


class Sigmoid(ScalarFunction):
    """Sigmoid function $f(x) = 1 / (1 + e^{-x})$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the sigmoid of the input."""
        sigmoid_value = operators.sigmoid(a)
        ctx.save_for_backward(sigmoid_value)
        return sigmoid_value

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the derivative of the sigmoid function with respect to the output gradient 'd_output'."""
        (sigmoid_value,) = ctx.saved_values
        return d_output * sigmoid_value * (1 - sigmoid_value)


class Log(ScalarFunction):
    """Log function $f(x) = log(x)$"""

    @staticmethod
    def forward(ctx: Context, a: float) -> float:
        """Compute the natural logarithm of the input."""
        ctx.save_for_backward(a)
        return operators.log(a)

    @staticmethod
    def backward(ctx: Context, d_output: float) -> float:
        """Compute the derivative of the log function with respect to the input value 'a' and the output gradient 'd_output'."""
        (a,) = ctx.saved_values
        return operators.log_back(a, d_output)


# To implement.


# TODO: Implement for Task 1.2.
