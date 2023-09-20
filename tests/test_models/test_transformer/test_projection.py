import pytest
import torch

from pydentification.models.transformer import LinearProjection


@pytest.mark.parametrize(
    ["n_input_time_steps", "n_input_state_variables", "n_output_time_steps", "n_output_state_variables"],
    (
        (10, 1, 10, 1),  # default settings
        (10, 4, 10, 1),  # down projection
        (10, 1, 10, 4),  # up projection
        (10, 4, 10, 4),  # high dimensional
    ),
)
def test_projection(
    n_input_time_steps: int,
    n_input_state_variables: int,
    n_output_time_steps: int,
    n_output_state_variables: int,
):
    module = LinearProjection(
        n_input_time_steps=n_input_time_steps,
        n_input_state_variables=n_input_state_variables,
        n_output_time_steps=n_output_time_steps,
        n_output_state_variables=n_output_state_variables,
    )

    inputs = torch.zeros((1, n_input_time_steps, n_input_state_variables))

    with torch.no_grad():
        outputs = module(inputs)  # type: ignore

    assert outputs.shape == (1, n_output_time_steps, n_output_state_variables)
