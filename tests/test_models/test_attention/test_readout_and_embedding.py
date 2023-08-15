from typing import Callable

import pytest
import torch

from pydentification.models.attention import LinearEmbedding, LinearReadout


@pytest.mark.parametrize(
    ["n_input_time_steps", "n_input_state_variables", "n_output_time_steps", "n_output_state_variables", "module"],
    (
        (10, 1, 10, 1, LinearEmbedding),  # default settings
        (10, 4, 10, 1, LinearEmbedding),  # down projection
        (10, 1, 10, 4, LinearEmbedding),  # up projection
        (10, 4, 10, 4, LinearEmbedding),  # high dimensional
        (10, 1, 10, 1, LinearReadout),  # default settings
        (10, 4, 10, 1, LinearReadout),  # down projection
        (10, 1, 10, 4, LinearReadout),  # up projection
        (10, 4, 10, 4, LinearReadout),  # high dimensional
    ),
)
def test_readout_and_embedding(
    n_input_time_steps: int,
    n_input_state_variables: int,
    n_output_time_steps: int,
    n_output_state_variables: int,
    module: Callable,
):
    module = module(
        n_input_time_steps=n_input_time_steps,
        n_input_state_variables=n_input_state_variables,
        n_output_time_steps=n_output_time_steps,
        n_output_state_variables=n_output_state_variables,
    )

    inputs = torch.zeros((1, n_input_time_steps, n_input_state_variables))

    with torch.no_grad():
        outputs = module(inputs)  # type: ignore

    assert outputs.shape == (1, n_output_time_steps, n_output_state_variables)
