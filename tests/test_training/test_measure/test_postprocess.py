import pytest
import torch
from torch import Tensor

from pydentification.training.measure.postprocess import make_tensor_lovely, tensor_describe


@pytest.mark.parametrize(
    ["tensor", "expected"],
    [
        (torch.tensor([1, 2, 3]), "tensor[3] i64 x∈[1, 3] μ=2.000 σ=1.000 [1, 2, 3]"),
        (
            torch.ones(10),
            "tensor[10] x∈[1.000, 1.000] μ=1.000 σ=0. [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]",  # noqa: E501
        ),
        (torch.ones(1000), "tensor[1000] 3.9Kb x∈[1.000, 1.000] μ=1.000 σ=0."),
    ],
)
def test_make_tensor_lovely(tensor: Tensor, expected: str):
    assert make_tensor_lovely(tensor) == expected


@pytest.mark.parametrize(
    ["tensor", "expected"],
    [
        (
            torch.tensor([1, 2, 3]),
            {"count": 3.0, "mean": 2.0, "std": 1.0, "min": 1.0, "25%": 1.5, "50%": 2.0, "75%": 2.5, "max": 3.0},
        ),
        (
            torch.ones(10),
            {"count": 10.0, "mean": 1.0, "std": 0.0, "min": 1.0, "25%": 1.0, "50%": 1.0, "75%": 1.0, "max": 1.0},
        ),
        (
            torch.ones(1000),
            {"count": 1000.0, "mean": 1.0, "std": 0.0, "min": 1.0, "25%": 1.0, "50%": 1.0, "75%": 1.0, "max": 1.0},
        ),
    ],
)
def test_tensor_describe(tensor: Tensor, expected: dict[str, float]):
    assert tensor_describe(tensor) == expected
