import pytest
import torch

from pydentification.models.networks.transformer import DelayLineFeedforward


@pytest.mark.parametrize(
    ["n_time_steps", "n_state_variables", "bias", "skip_connection"],
    (
        (10, 1, True, False),  # default settings
        (10, 1, False, False),  # no bias
        (10, 1, True, True),  # with skip connection
        (10, 4, True, False),  # 4 state variables
        (10, 4, False, False),  # 4 state variables with no bias
        (10, 4, True, True),  # 4 state variables with skip connection
    ),
)
def test_delay_line_feedforward(
    n_time_steps: int,
    n_state_variables: int,
    bias: bool,
    skip_connection: bool,
):
    module = DelayLineFeedforward(
        n_time_steps=n_time_steps,
        n_state_variables=n_state_variables,
        bias=bias,
        skip_connection=skip_connection,
    )

    inputs = torch.zeros((1, n_time_steps, n_state_variables))

    with torch.no_grad():
        outputs = module(inputs)  # type: ignore

    assert outputs.shape == inputs.shape  # this module does not change the shape of the input
