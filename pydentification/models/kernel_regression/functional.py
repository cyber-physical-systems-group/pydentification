import math
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


def kernel_regression(
    inputs: Tensor,
    memory: Tensor,
    targets: Tensor,
    kernel: Callable,
    bandwidth: float,
    p: int = 2,
    return_kernel_density: bool = False,
) -> Tensor:
    """
    Functional implementation of kernel regression, which computes the weighted average of the memory (training data)
    using kernel function given as callable.

    :note: This implementation only support MISO systems, i.e. multiple inputs, but only single output.

    :param inputs: points where function is to be predicted from training data, 2D tensors with shape (INPUT_SIZE, DIM)
    :param memory: points where function was evaluated during training, 2D tensors with shape (MEMORY_SIZE, DIM)
    :param targets: values of function at memory points, 1D tensor with shape (MEMORY_SIZE,)
    :param kernel: callable kernel function that takes memory and input points and returns kernel density as tensor
    :param bandwidth: bandwidth of the kernel
    :param p: exponent for point-wise distance, defaults to 2
    :param return_kernel_density: if True, returns kernel density for each input point as well
    """
    distances = torch.cdist(memory, inputs, p=p)  # (MEMORY_SIZE, INPUT_SIZE)
    kernels = kernel(distances, width=bandwidth, offset=float(0))  # (MEMORY_SIZE, INPUT_SIZE)
    # unsqueeze along last dimension to allow broadcasting target values over kernel density
    targets = targets.unsqueeze(-1)  # (MEMORY_SIZE, 1)
    # average over kernel density for each point function is predicted for
    predictions = (kernels * targets).sum(axis=0) / kernels.sum(dim=0)

    if return_kernel_density:
        return predictions, kernels

    return predictions


def kernel_regression_bounds(
    kernels: Tensor, dim: int, bandwidth: float, delta: float, lipschitz_constant: float, noise_variance: float
) -> Tensor:
    """
    Computes bounds for the error of kernel regression, see: https://arxiv.org/abs/2305.03295

    :param kernels: kernel density computed by the estimator, kernel needs to have a compact carrier
    :param dim: dimensionality of the output space
    :param bandwidth: bandwidth of the kernel, `h` in equations
    :param delta: confidence level, parameter set by the user, `delta` in equations
    :param lipschitz_constant: Lipschitz constant of the function being estimated, `L` in equations
    :param noise_variance: variance of the noise in the function being estimated, `sigma` in equations

    :return: symmetric bounds computed as difference from the predictions. With probability 1 - delta, the true
             function value is in the interval [prediction - bound, prediction + bound]
    """
    kappa = kernels.sum(dim=0)  # kernel density for each point
    # kappa <= 1 component of the alpha
    inner_const = math.pow(2, dim / 2) / delta
    lower = (kappa <= 1) * math.sqrt(math.log(inner_const))
    # kappa > 1 component of the alpha
    inner = torch.pow(1 + kappa, dim / 2) / delta
    upper = (kappa > 1) * torch.sqrt(kappa * torch.log(inner))
    alpha = lower + upper  # sum of masked components for bounds
    # Lh + 2 sigma_k alpha(x, delta) / kappa(x)
    bounds = lipschitz_constant * bandwidth + 2 * noise_variance * alpha / kappa

    return bounds
