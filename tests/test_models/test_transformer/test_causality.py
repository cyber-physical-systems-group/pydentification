from typing import Any

import pytest
import torch
from torch import Tensor

from pydentification.models.networks import transformer


def change_time_series_signal(signal: Tensor, change_step: int, value: float) -> Tensor:
    """Modify the signal to have a change at the given time step"""
    signal = signal.clone()

    if len(signal.shape) == 2:
        signal[:, change_step] = value
    else:
        signal[:, change_step, :] = value

    return signal


@pytest.mark.parametrize(
    ["kwargs", "shape", "change_step"],
    (
        ({"in_features": 10, "out_features": 10}, (1, 10), 5),  # batch_size = 1, n_time_steps = 10
        ({"in_features": 10, "out_features": 10, "bias": False}, (1, 10), 5),  # with bias
        ({"in_features": 10, "out_features": 10}, (5, 10), 5),  # with higher batch size
    ),
)
def test_masked_linear_causality(kwargs: dict[str, Any], shape: tuple, change_step: int):
    torch.manual_seed(42)  # needed for second assert
    module = transformer.MaskedLinear(**kwargs)

    with torch.no_grad():
        reference = module(torch.ones(shape))  # type: ignore
        # change at given time step to two
        changed = module(change_time_series_signal(torch.ones(shape), change_step=change_step, value=2.0))

        # check that the change is only at the given time step
        assert torch.allclose(reference[:, :change_step], changed[:, :change_step])
        # this partly relies on initialization, if weight for some connection is close to zero (random seed used)
        assert not torch.allclose(reference[:, change_step:], changed[:, change_step:])


@pytest.mark.parametrize(
    ["kwargs", "shape", "change_step"],
    (
        # batch_size = 1, n_time_steps = 10, n_state_variables = 1
        ({"n_time_steps": 10, "n_state_variables": 1}, (1, 10, 1), 5),
        ({"n_time_steps": 10, "n_state_variables": 1, "bias": False}, (1, 10, 1), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 1, "skip_connection": True}, (1, 10, 1), 5),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 1}, (5, 10, 1), 5),  # with higher batch size
        # batch_size = 1, n_time_steps = 10, n_state_variables = 5
        ({"n_time_steps": 10, "n_state_variables": 5}, (1, 10, 5), 5),
        ({"n_time_steps": 10, "n_state_variables": 5, "bias": False}, (1, 10, 5), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 5, "skip_connection": True}, (1, 10, 5), 5),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 5}, (5, 10, 5), 5),  # with higher batch size
        # batch_size = 1, n_time_steps = 10, n_state_variables = 5
        # change input early
        ({"n_time_steps": 10, "n_state_variables": 1}, (1, 10, 1), 1),
        ({"n_time_steps": 10, "n_state_variables": 1, "bias": False}, (1, 10, 1), 1),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 1, "skip_connection": True}, (1, 10, 1), 1),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 1}, (5, 10, 1), 1),  # with higher batch size
    ),
)
def test_masked_delay_line_causality(kwargs: dict[str, Any], shape: tuple, change_step: int):
    torch.manual_seed(42)  # needed for second assert
    module = transformer.CausalDelayLineFeedforward(**kwargs)
    module = transformer.DelayLineFeedforward(**kwargs)

    with torch.no_grad():
        reference = module(torch.ones(shape))  # type: ignore
        # change at given time step to two
        changed = module(change_time_series_signal(torch.ones(shape), change_step=change_step, value=2.0))

        # check that the change is only at the given time step
        assert torch.allclose(reference[:, :change_step], changed[:, :change_step])
        # this partly relies on initialization, if weight for some connection is close to zero (random seed used)
        assert not torch.allclose(reference[:, change_step:], changed[:, change_step:])


@pytest.mark.parametrize(
    ["kwargs", "shape", "change_step"],
    (
        # batch_size = 1, n_time_steps = 10, n_state_variables = 1
        ({"n_time_steps": 10, "n_state_variables": 1}, (1, 10, 1), 5),
        ({"n_time_steps": 10, "n_state_variables": 1, "bias": False}, (1, 10, 1), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 1, "skip_connection": True}, (1, 10, 1), 5),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 1}, (5, 10, 1), 5),  # with higher batch size
        # batch_size = 1, n_time_steps = 10, n_state_variables = 5
        ({"n_time_steps": 10, "n_state_variables": 5}, (1, 10, 5), 5),
        ({"n_time_steps": 10, "n_state_variables": 5, "bias": False}, (1, 10, 5), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 5, "skip_connection": True}, (1, 10, 5), 5),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 5}, (5, 10, 5), 5),  # with higher batch size
        # batch_size = 1, n_time_steps = 10, n_state_variables = 5
        # change input early
        ({"n_time_steps": 10, "n_state_variables": 1}, (1, 10, 1), 1),
        ({"n_time_steps": 10, "n_state_variables": 1, "bias": False}, (1, 10, 1), 1),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 1, "skip_connection": True}, (1, 10, 1), 1),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 1}, (5, 10, 1), 1),  # with higher batch size
    ),
)
def test_masked_delay_line_causality(kwargs: dict[str, Any], shape: tuple, change_step: int):
    torch.manual_seed(42)  # needed for second assert
    module = transformer.CausalDelayLineFeedforward(**kwargs)

    with torch.no_grad():
        reference = module(torch.ones(shape))  # type: ignore
        # change at given time step to two
        changed = module(change_time_series_signal(torch.ones(shape), change_step=change_step, value=2.0))

        # check that the change is only at the given time step
        assert torch.allclose(reference[:, :change_step], changed[:, :change_step])
        # this partly relies on initialization, if weight for some connection is close to zero (random seed used)
        assert not torch.allclose(reference[:, change_step:], changed[:, change_step:])


@pytest.mark.parametrize(
    ["kwargs", "shape", "change_step"],
    (
        # batch_size = 1, n_time_steps = 10, n_state_variables = 1
        ({"n_time_steps": 10, "n_state_variables": 1, "n_heads": 1}, (1, 10, 1), 5),
        ({"n_time_steps": 10, "n_state_variables": 1, "n_heads": 1, "bias": False}, (1, 10, 1), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 1, "n_heads": 1}, (5, 10, 1), 5),  # with higher batch size
        # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 1, "n_heads": 1, "skip_connection": True}, (1, 10, 1), 5),
        # batch_size = 1, n_time_steps = 10, n_state_variables = 5
        ({"n_time_steps": 10, "n_state_variables": 5, "n_heads": 1}, (1, 10, 5), 5),
        ({"n_time_steps": 10, "n_state_variables": 5, "n_heads": 1, "bias": False}, (1, 10, 5), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 5, "n_heads": 1}, (5, 10, 5), 5),  # with higher batch size
        # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 5, "n_heads": 1, "skip_connection": True}, (1, 10, 5), 5),
        # change input early batch_size = 1, n_time_steps = 10, n_state_variables = 1
        ({"n_time_steps": 10, "n_state_variables": 1, "n_heads": 1}, (1, 10, 1), 1),
        ({"n_time_steps": 10, "n_state_variables": 1, "n_heads": 1, "bias": False}, (1, 10, 1), 1),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 1, "n_heads": 1}, (5, 10, 1), 1),  # with higher batch size
        # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 1, "n_heads": 1, "skip_connection": True}, (1, 10, 1), 1),
        # with many heads and input early batch_size = 1, n_time_steps = 10, n_state_variables = 16
        ({"n_time_steps": 10, "n_state_variables": 16, "n_heads": 4}, (1, 10, 16), 5),
        ({"n_time_steps": 10, "n_state_variables": 16, "n_heads": 4, "bias": False}, (1, 10, 16), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 16, "n_heads": 4}, (5, 10, 16), 5),  # with higher batch size
        # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 16, "n_heads": 4, "skip_connection": True}, (1, 10, 16), 5),
    ),
)
def test_masked_self_attention_causality(kwargs: dict[str, Any], shape: tuple, change_step: int):
    torch.manual_seed(42)  # needed for second assert
    module = transformer.DynamicalSelfAttention(**dict(is_causal=True, **kwargs))  # only test causal variant

    with torch.no_grad():
        reference = module(torch.ones(shape))  # type: ignore
        # change at given time step to two
        changed = module(change_time_series_signal(torch.ones(shape), change_step=change_step, value=2.0))

        # check that the change is only at the given time step
        assert torch.allclose(reference[:, :change_step], changed[:, :change_step])
        # this partly relies on initialization, if weight for some connection is close to zero (random seed used)
        assert not torch.allclose(reference[:, change_step:], changed[:, change_step:])


@pytest.mark.parametrize(
    ["kwargs", "shape", "change_step"],
    (
        # batch_size = 1, n_time_steps = 10, n_state_variables = 1
        ({"n_time_steps": 10, "n_state_variables": 1}, (1, 10, 1), 5),
        ({"n_time_steps": 10, "n_state_variables": 1, "bias": False}, (1, 10, 1), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 1, "skip_connection": True}, (1, 10, 1), 5),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 1}, (5, 10, 1), 5),  # with higher batch size
        # batch_size = 1, n_time_steps = 10, n_state_variables = 5
        ({"n_time_steps": 10, "n_state_variables": 5}, (1, 10, 5), 5),
        ({"n_time_steps": 10, "n_state_variables": 5, "bias": False}, (1, 10, 5), 5),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 5, "skip_connection": True}, (1, 10, 5), 5),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 5}, (5, 10, 5), 5),  # with higher batch size
        # batch_size = 1, n_time_steps = 10, n_state_variables = 5
        # change input early
        ({"n_time_steps": 10, "n_state_variables": 1}, (1, 10, 1), 1),
        ({"n_time_steps": 10, "n_state_variables": 1, "bias": False}, (1, 10, 1), 1),  # with bias
        ({"n_time_steps": 10, "n_state_variables": 1, "skip_connection": True}, (1, 10, 1), 1),  # with skip connection
        ({"n_time_steps": 10, "n_state_variables": 1}, (5, 10, 1), 1),  # with higher batch size
    ),
)
def test_delay_line_causality(kwargs: dict[str, Any], shape: tuple, change_step: int):
    """Reference test for unmasked model, which is not causal"""
    torch.manual_seed(42)  # needed for second assert
    module = transformer.DelayLineFeedforward(**kwargs)

    with torch.no_grad():
        reference = module(torch.ones(shape))  # type: ignore
        # change at given time step to two
        changed = module(change_time_series_signal(torch.ones(shape), change_step=change_step, value=2.0))

        # check that the change is only at the given time step
        assert not torch.allclose(reference[:, :change_step], changed[:, :change_step])
        # this partly relies on initialization, if weight for some connection is close to zero (random seed used)
        assert not torch.allclose(reference[:, change_step:], changed[:, change_step:])
