from typing import Callable

import torch

from .feedforward import DelayLineFeedforward
from .self_attention import DynamicalSelfAttention


class DynamicalSelfAttentionBlock(torch.nn.Module):
    def __init__(
        self,
        n_time_steps: int,
        n_state_variables: int,
        n_heads: int,
        bias: bool = True,
        sa_skip_connection: bool = False,
        ff_skip_connection: bool = False,
        activation: Callable = torch.nn.functional.gelu,
    ):
        super(DynamicalSelfAttentionBlock, self).__init__()

        self.activation = activation
        self.block = torch.nn.Sequential(
            DynamicalSelfAttention(
                n_time_steps=n_time_steps,
                n_state_variables=n_state_variables,
                n_heads=n_heads,
                bias=bias,
                skip_connection=sa_skip_connection,
            ),
            DelayLineFeedforward(
                n_time_steps=n_time_steps,
                n_state_variables=n_state_variables,
                bias=bias,
                skip_connection=ff_skip_connection,
            ),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.activation(self.block(inputs))
