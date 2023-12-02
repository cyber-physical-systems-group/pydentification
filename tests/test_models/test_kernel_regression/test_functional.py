import pytest
import torch

from pydentification.models.kernel_regression.functional import point_wise_distance_tensor


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
