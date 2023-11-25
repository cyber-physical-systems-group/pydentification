import torch
from torch import Tensor


def bounded_linear_unit(inputs: Tensor, bounds: float | Tensor, inplace: bool = False) -> Tensor:
    """
    Bounded linear activation function. It means that the output is linear in range [-bounds, bounds] and clamped
    outside of it to the values of the bounds. Bounds can be scalar of tensor of the same shape as inputs.
    """
    out = inputs if inplace else None
    return torch.clamp(inputs, min=-bounds, max=bounds, out=out)


def universal_activation(inputs: Tensor, inplace: bool = False) -> Tensor:
    """
    Universal activation function, for which finite number of neurons can approximate any continuous function on
    d-dimensional hypercube. The number of neurons required is equal to 36d(2d + 1) in each feedforward layer and the
    number of needed layers is equal to 11. The proof is contained in the paper:

    :reference: https://www.jmlr.org/papers/volume23/21-1404/21-1404.pdf
    """

    def _lower(x: Tensor) -> Tensor:
        """Lower part of piecewise activation function applied to negative inputs."""
        return x / (torch.abs(x) + 1)

    def _upper(x: Tensor) -> Tensor:
        """Upper part of piecewise activation function applied to positive inputs."""
        floored = 2 * torch.floor((x + 1) / 2)
        return torch.abs(x - floored)

    outputs = _lower(inputs) * (inputs < 0) + _upper(inputs) * (inputs >= 0)

    if inplace:
        inputs.copy_(outputs)

    return outputs
