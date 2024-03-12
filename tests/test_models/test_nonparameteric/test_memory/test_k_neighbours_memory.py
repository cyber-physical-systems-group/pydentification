import pytest
import torch
from torch import Tensor

from pydentification.models.nonparametric.memory.abstract import MemoryManager


@pytest.mark.parametrize(
    "memory_manager",
    [
        pytest.lazy_fixture("nn_descent_memory_manager"),
        pytest.lazy_fixture("exact_memory_manager"),
        pytest.lazy_fixture("scikit_auto_memory_manager"),
        pytest.lazy_fixture("scikit_kd_tree_memory_manager"),
        pytest.lazy_fixture("scikit_ball_tree_memory_manager"),
    ],
)
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
def test_k_neighbours_query(memory_manager: MemoryManager, points: Tensor, k: int, expected: Tensor):
    memory, _, _ = memory_manager.query(points, k=k)  # type: ignore
    torch.testing.assert_close(memory, expected)
