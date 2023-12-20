import pytest
import torch
from torch import Tensor

from pydentification.models.nonparametric.memory import NNDescentMemoryManager, needs_tensor_with_dims


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
    def func(_, t: Tensor) -> bool:  # first argument is self of MemoryManager, which is not used
        return True  # dummy function to test decorator

    assert func(None, tensor)  # pass None as self and check if error was not raised


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
    def func(_, t: Tensor) -> bool:  # first argument is self of MemoryManager, which is not used
        return True  # dummy function to test decorator

    with pytest.raises(ValueError):
        func(None, tensor)  # pass None as self


@pytest.fixture(scope="module")
def linspace_nn_descent_memory_manager():
    memory = torch.linspace(0, 1, 101).unsqueeze(-1)  # 101 points in [0, 1] range spaced by 0.01 and shape [101, 1]
    targets = 2 * memory  # dummy targets
    return NNDescentMemoryManager(memory, targets, metric="euclidean")


@pytest.mark.parametrize(
    "points, k, expected",
    (
        # query for single point
        (torch.tensor([[0.5]]), 1, torch.tensor([[0.5]])),
        # query for multiple points
        (torch.tensor([[0.5]]), 3, torch.tensor([[0.49], [0.5], [0.51]])),
        # query on the edge of range
        (torch.tensor([[0.0]]), 1, torch.tensor([[0.0]])),
        (torch.tensor([[1.0]]), 1, torch.tensor([[1.0]])),
        # query for multiple points on the edge of range
        (torch.tensor([[0.0]]), 5, torch.tensor([[0.0], [0.01], [0.02], [0.03], [0.04]])),
        (torch.tensor([[1.0]]), 5, torch.tensor([[0.96], [0.97], [0.98], [0.99], [1.0]])),
        # batch query
        (torch.tensor([[0.5], [0.25]]), 1, torch.tensor([[0.25], [0.5]])),
        # batch query with multiple points
        (torch.tensor([[0.5], [0.25]]), 3, torch.tensor([[0.24], [0.25], [0.26], [0.49], [0.5], [0.51]])),
        # batch query with multiple points without exact matches to memory
        (torch.tensor([[0.501], [0.2501]]), 3, torch.tensor([[0.24], [0.25], [0.26], [0.49], [0.5], [0.51]])),
    ),
)
def test_nn_descent_memory_manager(points: Tensor, k: int, expected: Tensor, linspace_nn_descent_memory_manager):
    # query with high epsilon to get certain results
    memory, _ = linspace_nn_descent_memory_manager.query_nearest(points, k, epsilon=1.0)  # ignore targets
    torch.testing.assert_close(memory, expected)
