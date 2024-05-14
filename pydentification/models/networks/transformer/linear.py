import math

import torch
from torch import Tensor, nn


class LinearProjection(nn.Module):
    """
    Module converting time series with shape (batch_size, n_input_time_steps, n_input_state_variables) into time series
    with shape (batch_size, n_output_time_steps, n_output_state_variables) using learned linear transformation.

    It can be used to up or down project state variables and shorten or extend time series length.
    """

    def __init__(
        self,
        n_input_time_steps: int,
        n_output_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
        bias: bool = True,
    ):
        """
        :param n_input_time_steps: number of time steps in the input signal
        :param n_output_time_steps: number of time steps to produce after linear operation
        :param n_input_state_variables: number of state input variables
        :param n_output_state_variables: number of states to produce
        :param bias: if True bias will be used in linear operation
        """
        super(LinearProjection, self).__init__()

        self.n_input_time_steps = n_input_time_steps
        self.n_output_time_steps = n_output_time_steps
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables

        self.flatten = nn.Flatten(start_dim=1)
        self.projection = nn.Linear(
            in_features=self.n_input_time_steps * self.n_input_state_variables,
            out_features=self.n_output_time_steps * self.n_output_state_variables,
            bias=bias,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        batch_size = inputs.size(0)
        variables = self.flatten(inputs)
        variables = self.projection(variables)
        outputs = torch.reshape(variables, shape=(batch_size, self.n_output_time_steps, self.n_output_state_variables))
        return outputs


class MaskedLinear(nn.Module):
    """
    This module is a linear layer with a mask applied to the weights. The mask is a lower triangular matrix, so when
    delay-line measurements of time-series are passed through this layer, the output will be a causal prediction, with
    connections only allowed from future to past time-steps.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        """
        For parameters docstrings see nn.Linear
        source: https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
        """
        super(MaskedLinear, self).__init__()

        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        # following implementation from nn.Linear
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, inputs: Tensor) -> Tensor:
        # lower triangular mask with ones in elements where connection is allowed
        mask = torch.tril((torch.full((self.in_features, self.in_features), 1.0)), diagonal=0)
        mask = mask.to(self.weight.device).to(self.weight.dtype)

        return nn.functional.linear(inputs, self.weight * mask, self.bias)

    def extra_repr(self) -> str:
        return f"masked_in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"
