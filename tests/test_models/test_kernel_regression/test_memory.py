import pytest
import torch
from torch import Tensor

from pydentification.models.kernel_regression.memory import needs_tensor_with_dims


@pytest.mark.parametrize(
    "tensor, dims",
    [
        (torch.zeros([2, 2]), (2, 2)),
        (torch.zeros([5, 5]), (5, None)),
        (torch.zeros([1, 1]), (1, None)),
        (torch.zeros([5, 1]), (None, 1)),
        (torch.zeros([2, 2]), (None, None)),
        (torch.zeros([2, 2, 2]), (2, 2, 2)),
    ],
)
def test_needs_tensor_with_dims_ok(tensor: Tensor, dims: tuple[int | None]):
    @needs_tensor_with_dims(*dims)
    def func(_: Tensor) -> bool:
        return True  # dummy function to test decorator

    assert func(tensor)  # check if error was not raised


@pytest.mark.parametrize(
    "tensor, dims",
    [
        (torch.zeros([10, 1, 1]), (None, None)),
        (torch.zeros([1, 10, 1]), (None, None)),
        (torch.zeros([10, 1]), (10, 2)),
        (torch.zeros([10, 1]), (2, 10)),
        (torch.zeros([10, 1]), (None, 2)),
    ],
)
def test_needs_tensor_with_dims_not_ok(tensor: Tensor, dims: tuple[int | None]):
    @needs_tensor_with_dims(*dims)
    def func(_: Tensor) -> bool:
        return True  # dummy function to test decorator

    with pytest.raises(ValueError):
        func(tensor)
