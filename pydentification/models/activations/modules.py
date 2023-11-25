from torch import Tensor
from torch.nn import Module

from . import functional as func


class BoundedLinearUnit(Module):
    """
    Bounded linear activation function. It means that the output is linear in range [-bounds, bounds] and clamped
    outside of it to the values of the bounds. Bounds can be scalar of tensor of the same shape as inputs.
    """

    def __init__(self, bounds: float | Tensor | None = None):
        """
        :param bounds: bounds for the linear unit, can be scalar or tensor of the same shape as inputs
                       bounds given in __init__ are static, applied irrespective of the input bounds
        """
        super(BoundedLinearUnit, self).__init__()
        self.static_bounds = bounds

    def forward(self, inputs: Tensor, bounds: float | Tensor | None = None) -> Tensor:
        bounds = bounds if bounds is not None else self.static_bounds
        return func.bounded_linear_unit(inputs, bounds)


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
