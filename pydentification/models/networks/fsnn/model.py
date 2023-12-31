from functools import cached_property

import torch
from torch import Tensor
from torch.nn import Module

from pydentification.models.modules.feedforward import TimeSeriesLinear
from pydentification.models.modules.orthogonal.fourier import IRFFTModule, RFFTModule


class FrequencyLinear(Module):
    def __init__(
        self,
        n_input_time_steps: int,
        n_output_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
        param_dtype: torch.dtype = torch.cfloat,
        output_dtype: torch.dtype = torch.float32,
        use_bias: bool = True,
    ):
        """
        :param n_input_time_steps: number of time steps in the input signal
        :param n_output_time_steps: number of time steps to produce after linear operation in frequency space
        :param n_input_state_variables: number of state input variables, each state is processed using 1D FFT
        :param n_output_state_variables: number of states to produce
        :param param_dtype: parameter type of linear layer operating on frequency space of the signal
                            can be complex or real, but for real torch will autocast frequency space to real numbers
                            which causes loss of information, defaults to cfloat
        :param output_dtype: output parameter type, should be real data type, defaults to float32
        :param use_bias: if True network used bias in linear operation
        """
        super(FrequencyLinear, self).__init__()

        self.n_input_time_steps = n_input_time_steps
        self.n_output_time_steps = n_output_time_steps
        self.n_input_state_variables = n_input_state_variables
        self.n_output_state_variables = n_output_state_variables

        self.param_dtype = param_dtype
        self.output_dtype = output_dtype

        self.stack = torch.nn.Sequential(
            RFFTModule(),
            TimeSeriesLinear(
                n_input_time_steps=self.input_fourier_modes,
                n_output_time_steps=self.output_fourier_modes,
                n_input_state_variables=self.n_input_state_variables,
                n_output_state_variables=self.n_output_state_variables,
                dtype=self.param_dtype,
                use_bias=use_bias,
            ),
            IRFFTModule(),
        )

    @cached_property
    def input_fourier_modes(self):
        """Number of frequency modes resulting from RFFT"""
        return self.n_input_time_steps // 2 + 1

    @cached_property
    def output_fourier_modes(self):
        """
        Return number of frequency models required to produce time-series with desired length
        When self.n_output_time_steps is odd, produces one more and slices it off,
        since IRFFT always return even number of samples
        """
        if self.n_output_time_steps % 2 != 0:
            return self.n_output_time_steps // 2 + 2  # IRFFT produces self.n_output_time_steps + 1 samples
        return self.n_output_time_steps // 2 + 1

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = self.stack(inputs)
        # only slices when self.n_output_time_steps is odd
        return outputs.to(self.output_dtype)[:, : self.n_output_time_steps, :]


class TimeFrequencyLinear(Module):
    def __init__(
        self,
        n_input_time_steps: int,
        n_output_time_steps: int,
        n_input_state_variables: int,
        n_output_state_variables: int,
        dtype: torch.dtype = torch.float32,
    ):
        """
        :param n_input_time_steps: number of time steps in the input signal
        :param n_output_time_steps: number of time steps to produce after linear operation in frequency space
        :param n_input_state_variables: number of state input variables, each state is processed using 1D FFT
        :param n_output_state_variables: number of states to produce
        :param dtype: output parameter type, should be real data type, defaults to float32
        """
        super(TimeFrequencyLinear, self).__init__()

        self.frequency_linear = FrequencyLinear(
            n_input_time_steps=n_input_time_steps,
            n_output_time_steps=n_output_time_steps,
            n_input_state_variables=n_input_state_variables,
            n_output_state_variables=n_output_state_variables,
            output_dtype=dtype,
        )

        self.linear = TimeSeriesLinear(
            n_input_time_steps=n_input_time_steps,
            n_output_time_steps=n_output_time_steps,
            n_input_state_variables=n_input_state_variables,
            n_output_state_variables=n_output_state_variables,
            dtype=dtype,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        frequency_outputs = self.frequency_linear(inputs)  # type: ignore
        linear_outputs = self.linear(inputs)  # type: ignore

        return frequency_outputs + linear_outputs
