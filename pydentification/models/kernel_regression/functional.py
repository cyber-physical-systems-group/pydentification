from typing import Callable

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


def kernel_regression(inputs: Tensor, memory: Tensor, targets: Tensor, kernel: Callable) -> Tensor:
    """
    Functional implementation of kernel regression, which computes the weighted average of the memory (training data)
    using kernel function given as callable.

    :note: This implementation only support MISO systems, i.e. multiple inputs, but only single output.

    :param inputs: points where function is to be predicted from training data
    :param memory: points where function was evaluated during training
    :param targets: values of function at memory points
    :param kernel: callable kernel function that takes memory and input points and returns kernel density as tensor
    """
    kernels = kernel(memory, inputs)  # (MEMORY_SIZE, INPUT_SIZE)
    # unsqueeze along last dimension to allow broadcasting target values over kernel density
    targets = targets.unsqueeze(-1)  # (MEMORY_SIZE, 1)
    # average over kernel density for each point function is predicted for
    predictions = (kernels * targets).sum(axis=0) / kernels.sum(dim=0)

    return predictions
