import pytest
import torch
from torch import Tensor

from pydentification.models.nonparametric.memory.transformations import TruncateDelayLine


def batch_arange(start: int, end: int, batch_size: int = 1) -> Tensor:
    """Return batch of arange tensors from start to end for given batch size, always as float32"""
    return torch.arange(start, end).unsqueeze(dim=0).repeat(batch_size, 1).to(torch.float32)


@pytest.mark.parametrize(["points", "target_dim", "expected"], [
    # return last element for single item batch of points
    (batch_arange(1, 11), 1, torch.Tensor([[10]])),
    # return last 3 elements for single item batch of points
    (batch_arange(1, 11), 3, batch_arange(8, 11)),
    # return last 5 elements for single item batch of points
    (batch_arange(1, 11), 5, batch_arange(6, 11)),
])
def test_truncate_delay_line(points: Tensor, target_dim: int, expected: Tensor):
    transform = TruncateDelayLine(target_dim)
    # memory and query undergo the same transformation of keeping last `target_dim` elements
    memory = transform.before_prepare(points)
    query = transform.before_query(points)

    assert memory.size(-1) == target_dim
    assert query.size(-1) == target_dim
    torch.testing.assert_close(memory, expected)
    torch.testing.assert_close(query, expected)
