import math

import pytest
import torch
from torch import Tensor

from pydentification.data import decay, unbatch
from tests.test_data.utils import tensor_batch_iterable


@pytest.mark.parametrize(
    ["x", "gamma", "expected"],
    [
        # decay with gamma=0 should return original tensor
        (torch.ones(10), float(0), torch.ones(10)),
        # decay with gamma=0 applied to N-D tensor should return original tensor
        (torch.ones((10, 1)), float(0), torch.ones((10, 1))),
        # decay with gamma=1 should return elements of tensor multiplied by 1/e in each step
        (torch.ones(5), float(1), torch.Tensor([1, 1 / math.e, 1 / math.e**2, 1 / math.e**3, 1 / math.e**4])),
        # decay with gamma=1 applied to N-D tensor should return elements of tensor multiplied by 1/e in each step
        (
            torch.ones((1, 5)),
            float(1),
            torch.Tensor(
                [
                    [1, 1 / math.e, 1 / math.e**2, 1 / math.e**3, 1 / math.e**4],
                ]
            ),
        ),
    ],
)
def test_decay(x: Tensor, gamma: float, expected: Tensor):
    assert torch.allclose(decay(x, gamma=gamma), expected)


@pytest.mark.parametrize(
    ["n_batches", "batch_size", "batch_shape", "n_tensors", "expected_shape"],
    [
        # single batch with 32 items with shape (10, 1)
        (1, 32, (10, 1), 1, (32, 10, 1)),
        # single batch with 32 items with shape (10,) for features and targets
        (1, 32, (10,), 2, (32, 10)),
        # 10 batches with 32 items with shape (10, 1)
        (10, 32, (10, 1), 1, (320, 10, 1)),
        # 10 batches with 32 items with shape (10,) for features and targets
        (10, 32, (10,), 2, (320, 10)),
        # single item batches
        (10, 1, (10, 1), 1, (10, 10, 1)),
        # 4-tuple dataloader
        (10, 32, (10, 1), 4, (320, 10, 1)),
    ],
)
def test_unbatch(n_batches: int, batch_size: int, batch_shape: tuple, n_tensors: int, expected_shape: tuple):
    iterable = tensor_batch_iterable(n_batches, batch_size, batch_shape, n_tensors)

    for tensor in unbatch(iterable):
        assert tensor.shape == expected_shape  # in the test all input tensors have the same shape
