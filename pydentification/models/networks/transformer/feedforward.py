from typing import Callable

import torch
from torch import Tensor, nn

from .linear import MaskedLinear


class DelayLineFeedforward(nn.Module):
    """
    Linear transformation of the input signal using delay line, which means single row of weight matrix is multiplied
    with each time step of the input signal. For MIMO systems, different weight matrix is applied for each dimension.

    MIMO is implemented with flattening the input for better performance.
    """

    def __init__(self, n_time_steps: int, n_state_variables: int, bias: bool = True, skip_connection: bool = False):
        """
        :param n_time_steps: number of input and output time steps (they must be equal for self-attention)
        :param n_state_variables: number of state variables in the system or inner representation in the model
        :param bias: if True bias will be added to the linear module
        :param skip_connection: if True skip connection will be added to the output
        """
        super(DelayLineFeedforward, self).__init__()

        self.n_time_steps = n_time_steps
        self.n_state_variables = n_state_variables
        self.bias = bias
        self.skip_connection = skip_connection

        self.flatten = nn.Flatten()
        self.feedforward = nn.Linear(
            in_features=self.n_time_steps * self.n_state_variables,
            out_features=self.n_time_steps * self.n_state_variables,
            bias=self.bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        variables = self.flatten(inputs)  # flatten time steps
        variables = self.feedforward(variables)
        outputs = torch.reshape(variables, shape=inputs.shape)

        if self.skip_connection:
            outputs = inputs + outputs

        return outputs


class CausalDelayLineFeedforward(nn.Module):
    """
    Linear transformation of the input signal using delay line with casual mask, allowing only connections from future
    to past time steps, which means single row of weight matrix is multiplied with each time step of the input signal.

    For MIMO systems, different weight matrix is applied for each dimension, where each of the matrices is stored in
    MaskedLinear module in torch.nn.ModuleList.
    """

    def __init__(self, n_time_steps: int, n_state_variables: int, bias: bool = True, skip_connection: bool = False):
        """
        For parameters docstrings see pydentification.models.transformer.feedforward.DelayLineFeedforward
        """
        super(CausalDelayLineFeedforward, self).__init__()

        self.n_time_steps = n_time_steps
        self.n_state_variables = n_state_variables
        self.bias = bias
        self.skip_connection = skip_connection

        self.masked_modules = torch.nn.ModuleList(  # modules is a reserved name in torch.nn.Module
            [
                MaskedLinear(in_features=n_time_steps, out_features=n_time_steps, bias=bias)
                for _ in range(n_state_variables)
            ]
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.n_state_variables != inputs.shape[-1]:
            raise RuntimeError("Number of state variables must be equal to number of input channels!")

        # input and output of each module from list has shape = (batch_size, n_time_steps, 1)
        # stack outputs to get shape = (batch_size, n_time_steps, n_state_variables)
        outputs = torch.stack([module(inputs[:, :, state]) for state, module in enumerate(self.masked_modules)], dim=-1)

        if self.skip_connection:
            outputs = outputs + inputs

        return outputs


class TransformerFeedforward(torch.nn.Module):
    """
    This module contains feedforward network in transformer style for dynamical systems. It uses two fully connected
    layers with activation function between them, but instead of applying them point-wise, like in classical
    transformers, it uses the delay-line in both of them to apply different weights for each time step.
    """

    def __init__(
        self,
        n_time_steps: int,
        n_state_variables: int,
        hidden_dimension: int,
        activation: Callable = torch.nn.functional.gelu,
        bias: bool = True,
        skip_connection: bool = False,
    ):
        """
        :param n_time_steps: number of input and output time steps (they must be equal for self-attention)
        :param n_state_variables: number of state variables in the system or inner representation in the model
        :param hidden_dimension: dimension of the hidden layer between two fully connected layers
        :param activation: activation function used between two fully connected layers (use torch.nn.functional)
        :param bias: if True bias will be added to the linear module
        :param skip_connection: if True skip connection will be added to the output
        """
        super(TransformerFeedforward, self).__init__()

        self.n_time_steps = n_time_steps
        self.n_state_variables = n_state_variables
        self.hidden_dimension = hidden_dimension
        self.bias = bias
        self.skip_connection = skip_connection

        self.flatten = nn.Flatten()
        self.activation = activation

        self.up_projection = nn.Linear(
            in_features=self.n_state_variables * self.n_time_steps,  # variables and time-steps are flattened into one
            out_features=self.hidden_dimension,  # hidden dimension contains mixed state variables and time-steps
            bias=self.bias,
        )

        self.down_projection = nn.Linear(
            in_features=self.hidden_dimension,
            # output will be reshaped to variables and time-steps
            out_features=self.n_state_variables * self.n_time_steps,
            bias=self.bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        variables = self.flatten(inputs)  # flatten time steps
        variables = self.up_projection(variables)
        variables = self.activation(variables)
        variables = self.down_projection(variables)
        outputs = torch.reshape(variables, shape=inputs.shape)

        if self.skip_connection:
            outputs = inputs + outputs

        return outputs
