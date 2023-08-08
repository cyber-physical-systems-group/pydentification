from typing import Callable

import numpy as np
import pytest
import torch

from pydentification.models.orthogonal import fourier


@pytest.fixture
def time_array():
    return np.linspace(0, 4 * np.pi, 100, endpoint=False)


def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Counts number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.parametrize(
    "function_set",
    [
        ({np.sin}),
        ({np.sin, np.cos}),
        {np.sin, np.cos, lambda t: np.sin(2 * t)},
        {lambda t: t * 2},
        {lambda t: np.zeros(t.shape)},
    ],
)
def test_real_fourier_transforms(function_set: set[Callable], time_array):
    """
    Test checks if applying forward and backward Fourier transforms (implemented as torch.modules)
    returns the same input with precision up to 3 decimal places
    """
    inputs = np.column_stack([f(time_array) for f in function_set])
    inputs = torch.from_numpy(inputs).unsqueeze(dim=0)  # convert to batch with single tensor

    model = torch.nn.Sequential(
        fourier.RFFTModule(),
        fourier.IRFFTModule(),
    )

    outputs = model(inputs)
    np.testing.assert_almost_equal(inputs.numpy().squeeze(), outputs.numpy().squeeze(), decimal=3)


@pytest.mark.parametrize("model", [fourier.RFFTModule, fourier.CFFTModule, fourier.IRFFTModule])
def test_non_trainable(model: torch.nn.Module):
    """Tests if transforms implemented as modules are not-trainable"""
    assert count_trainable_parameters(model()) == 0  # type: ignore
