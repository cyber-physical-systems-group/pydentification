import pytest
import torch

from pydentification.models.nonparametric.functional import (
    kernel_regression,
    kernel_regression_bounds,
    point_wise_distance_tensor,
)
from pydentification.models.nonparametric.kernels import box_kernel


@pytest.mark.parametrize(
    "x, y, p, expected",
    [
        # input is batch of 3 vectors of size 1 with all ones, so distance matrix should be all zeros
        # the same arrays with direct computation and L1 norm
        (torch.ones([3, 1]), torch.ones([3, 1]), 1.0, torch.zeros([3, 3])),
        (torch.ones([3, 1]), torch.ones([3, 1]), 2.0, torch.zeros([3, 3])),  # L2 norm
        (torch.ones([3, 1]), torch.ones([3, 1]), 3.0, torch.zeros([3, 3])),  # L3 norm
        # input is batch of 3 vectors of size 1 with all ones for x and all zeros for y
        # distance matrix should be all ones for any norm
        (torch.ones([3, 1]), torch.zeros([3, 1]), 1.0, torch.ones([3, 3])),
        (torch.ones([3, 1]), torch.zeros([3, 1]), 2.0, torch.ones([3, 3])),
        (torch.ones([3, 1]), torch.zeros([3, 1]), 3.0, torch.ones([3, 3])),
        # input is batch of 3 vectors of size 1 with different values
        (
            torch.tensor([[1.0], [2.0], [3.0]]).to(torch.float32),
            torch.tensor([[1.0], [2.0], [3.0]]).to(torch.float32),
            1.0,
            torch.as_tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]]).unsqueeze(dim=-1).to(torch.float32),
        ),
        # input is batch of 2 vectors with 2 dimensions each
        # result is 2x2x2 tensor with point-wise distances for each dimension and each point
        (
            torch.tensor([[1.0, 1.0], [2.0, 2.0]]).to(torch.float32),
            torch.tensor([[2.0, 2.0], [3.0, 3.0]]).to(torch.float32),
            1.0,
            torch.as_tensor([[[1, 1], [2, 2]], [[0, 0], [1, 1]]]).to(torch.float32),
        ),
    ],
)
def test_point_wise_distance(x, y, p, expected):
    result = point_wise_distance_tensor(x, y, p)  # type: ignore
    assert torch.allclose(result, expected)


@pytest.mark.parametrize(
    ["inputs", "memory", "targets", "expected"],
    (
        # test case with constant function and all estimation points exactly matching memory
        (
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.ones(10).to(torch.float32),
            torch.ones(10).to(torch.float32),
        ),
        # test case with linear monotonic function f(x) = x and all estimation points exactly matching memory
        (
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.linspace(-1, 1, 10).to(torch.float32),
            torch.linspace(-1, 1, 10).to(torch.float32),
        ),
        # test case with linear monotonic function f(x) = - x and all estimation points exactly matching memory
        (
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            -torch.linspace(-1, 1, 10).to(torch.float32),
            -torch.linspace(-1, 1, 10).to(torch.float32),
        ),
    ),
)
def test_kernel_regression(inputs, memory, targets, expected):
    predictions = kernel_regression(inputs, memory, targets, box_kernel, bandwidth=0.1, p=2)
    assert torch.allclose(predictions, expected)


@pytest.mark.parametrize(
    ["inputs", "memory", "targets", "lc", "expected"],
    (
        # test case with constant function and all estimation points exactly matching memory
        (
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.ones(10).to(torch.float32),
            0,  # constant function has zero lipschitz constant
            torch.zeros(10).to(torch.float32),
        ),
        # test case with linear monotonic function f(x) = x and all estimation points exactly matching memory
        (
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.linspace(-1, 1, 10).to(torch.float32),
            1,  # linear function has lipschitz constant equal to absolute value of its slope
            # 1 / 10 is the deterministic parts of the bound
            # Lh = 1 * 1 / 10 = 1 / 10
            1 / 10 * torch.ones(10).to(torch.float32),
        ),
        # test case with linear monotonic function f(x) = - x and all estimation points exactly matching memory
        (
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            torch.linspace(-1, 1, 10).unsqueeze(dim=-1).to(torch.float32),
            -torch.linspace(-1, 1, 10).to(torch.float32),
            1,  # linear function has lipschitz constant equal to absolute value of its slope, abs(-1) = 1
            1 / 10 * torch.ones(10).to(torch.float32),  # for slope = -1 the same computation applies as for +1
        ),
    ),
)
def test_kernel_regression_bounds(inputs, memory, targets, lc, expected):
    h = 0.1  # bandwidth
    predictions, kernels = kernel_regression(
        inputs, memory, targets, box_kernel, bandwidth=h, p=2, return_kernel_density=True
    )
    bounds = kernel_regression_bounds(
        kernels,
        dim=1,  # functional kernel regression is only implemented for 1D outputs (MISO systems)
        bandwidth=h,
        delta=0.1,  # set by the user
        lipschitz_constant=lc,  # depends on the function being estimated, in test-cases this is known
        noise_variance=0.0,  # no noise in the unittest example
    )

    assert torch.allclose(bounds, expected)
    assert bounds.shape == (len(inputs),)
