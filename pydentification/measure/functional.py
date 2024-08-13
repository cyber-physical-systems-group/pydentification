import torch
from torch import Tensor


def conjugate_transpose(matrix: Tensor) -> Tensor:
    """Return conjugate transpose of complex matrix"""
    return torch.conj(matrix).transpose(-2, -1)
