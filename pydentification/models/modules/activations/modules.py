from torch import Tensor
from torch.nn import Module

from . import functional as func


class BoundedLinearUnit(Module):
    """
    Bounded linear activation function. It means that the output is linear in range [-bounds, bounds] and clamped
    outside of it to the values of the bounds. Bounds can be scalar of tensor of the same shape as inputs.
    """

    def __init__(
        self,
        lower: float | Tensor | None = None,
        upper: float | Tensor | None = None,
    ):
        """
        Bounds given in __init__ are static, applied irrespective of the input bounds
        they can be scalar or tensor of the same shape as inputs
        """
        super(BoundedLinearUnit, self).__init__()

        self.static_lower_bound = lower
        self.static_upper_bound = upper

    def forward(self, inputs: Tensor, bounds: float | Tensor | None = None) -> Tensor:
        lower = bounds if self.static_lower_bound is None else self.static_lower_bound
        upper = bounds if self.static_upper_bound is None else self.static_upper_bound

        return func.bounded_linear_unit(inputs, lower=lower, upper=upper)


class UniversalActivation(Module):
    """
    Universal activation function, for which finite number of neurons can approximate any continuous function on
    d-dimensional hypercube. The number of neurons required is equal to 36d(2d + 1) in each feedforward layer and the
    number of needed layers is equal to 11. The proof is contained in the paper:

    :reference: https://www.jmlr.org/papers/volume23/21-1404/21-1404.pdf
    """

    def __init__(self):
        super(UniversalActivation, self).__init__()

    def forward(self, inputs: Tensor) -> Tensor:
        return func.universal_activation(inputs)
