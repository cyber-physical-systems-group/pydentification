import pytest
import torch

from pydentification.models.networks.transformer import PositionalEncoding


@pytest.mark.parametrize(
    ["n_time_steps", "n_state_variables"],
    [
        (10, 1),
        (10, 8),
        (1, 8),
        (1, 1),
        (100, 100),
        (1000, 1000),
        (10, 1000),
        (1000, 10),
    ],
)
def test_positional_encoding(n_time_steps: int, n_state_variables: int):
    inputs = torch.randn(32, n_time_steps, n_state_variables)  # batch size of 32
    model = PositionalEncoding(n_state_variables=n_state_variables, n_time_steps=n_time_steps)

    outputs = model(inputs)
    assert outputs.shape == inputs.shape
