import torch
from torch import Tensor


def point_wise_distance_tensor(x: Tensor, y: Tensor, p: float = 1.0) -> Tensor:
    """
    Computes point-wise L_p distance matrix between x and y using given method. It does not reduce the dimensionality
    of the inputs, but returns a tensor (or matrix, if input is 1D) for each dimensions of each pair of points.

    :param x: 2D Input tensors should have shape (BATCH, DIMENSIONS)
    :param y: 2D Input tensors should have shape (BATCH, DIMENSIONS)
    :param p: exponent, defaults to 1.0
    """
    if p == 1.0:
        return torch.abs(x.unsqueeze(dim=1) - y.unsqueeze(dim=0))  # shortcut for P=1
    # generic equation for point-wise P-norm
    return torch.pow(torch.pow(torch.abs(x.unsqueeze(dim=1) - y.unsqueeze(dim=0)), p), 1 / p).squeeze()
