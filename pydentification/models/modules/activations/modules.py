from torch import Tensor
from torch.nn import Module

from . import functional as func


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
