import pytest
import torch
from torch.nn import Module

from pydentification.measure.parameters import SupportedNorms, parameter_norm


@pytest.fixture(scope="function")
def mock_module_parameters(request):
    """Fixture returning mocked torch module, containing dict given in request.param as named_parameters."""
    module = torch.nn.Module()
    module.named_parameters = lambda: request.param.items()
    return module


@pytest.mark.parametrize(
    ["mock_module_parameters", "weight_ord", "bias_ord", "expected"],
    [
        ({"weight": torch.ones(4, 4)}, None, None, {"weight": float(4)}),  # frobenius norm is used for matrices
        ({"weight": torch.ones(4, 4)}, 1, None, {"weight": float(4)}),  # sum of absolute values
        ({"weight": torch.ones(4, 4)}, 2, None, {"weight": float(4)}),  # largest singular value
        # different norms for weights and biases
        ({"weight": torch.ones(4, 4), "bias": torch.ones(4)}, 1, 2, {"weight": float(4), "bias": float(2)}),
        ({}, None, None, {}),  # empty module
    ],
    indirect=["mock_module_parameters"],
)
def test_parameter_norm(
    mock_module_parameters: Module, weight_ord: SupportedNorms, bias_ord: SupportedNorms, expected: dict[str, float]
):
    assert parameter_norm(mock_module_parameters, weight_ord, bias_ord) == expected
