import pytest
import torch

from pydentification.models.kernel_regression import kernels


@pytest.mark.parametrize("kernel", kernels.COMPACT_CARRIER_KERNELS)
def test_compact_carrier_kernels(kernel):
    """
    Test if kernels stored as compact carrier have no values in range (-2, -1) and (1, 2).
    Compact carrier kernels can only have values in range [-1, 1] when used with bandwidth=1.
    """
    eps = 1e-6  # small numerical offset
    low = torch.linspace(-2, -1 - eps, 100)
    high = torch.linspace(1 + eps, 2, 100)

    assert kernel(low, width=1, offset=0).sum() == 0
    assert kernel(high, width=1, offset=0).sum() == 0
