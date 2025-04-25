from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module


class RFFTModule(Module):
    """
    Module computes Fourier Transform for real input.

    It processes tensors with shape (BATCH, TIME_STEPS, SYSTEM_DIMENSIONS). For processing inputs of system with single
    state dimension pass unsqueezed Tensor with shape: (BATCH, TIME_STEPS, 1).
    """

    def __init__(
        self,
        n_time_steps: int | None = None,
        implementation: Literal["rfft", "matmul"] = "rfft",
        norm: str | None = None,
        dtype: torch.dtype = torch.cfloat,
    ):
        """
        :param n_time_steps: number of time to produce steps, see torch.fft.rfft for details
        :param implementation: implementation of RFFT, "rfft" uses torch.fft.rfft, "matmul" uses matrix multiplication
        :param norm: norm of RFFT, see torch.fft.rfft for details
        :param dtype: output data type, defaults to cfloat and should be complex
                      otherwise will cause loss of information after the RFFT transform
        """
        super(RFFTModule, self).__init__()

        self.n_input_time_steps = n_time_steps
        self.implementation = implementation
        self.norm = norm
        self.dtype = dtype

        self.requires_grad_(False)

        if self.implementation == "matmul":
            self.register_buffer("fft_matrix", self.build_fft_matrix())

    def build_fft_matrix(self) -> Tensor:
        """Precompute FFT matrix for matmul implementation."""
        return torch.fft.fft2(torch.eye(self.n_input_time_steps, dtype=self.dtype))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.implementation == "rfft":
            outputs = torch.fft.rfft(inputs, n=self.n_input_time_steps, norm=self.norm, dim=1)
        else:  # implementation == "matmul"
            outputs = torch.matmul(self.fft_matrix, inputs.to(self.fft_matrix.dtype))

        return outputs.to(self.dtype)


class IRFFTModule(Module):
    """
    Module computes Inverse Fourier Transform converting to real output.

    It processes tensors with shape (BATCH, FREQUENCY_MODELS, SYSTEM_DIMENSIONS). For processing inputs of system with
    single state dimension pass unsqueezed Tensor with shape: (BATCH, FREQUENCY_MODELS, 1).
    """

    def __init__(
        self,
        n_time_steps: int | None = None,
        implementation: Literal["rfft", "matmul"] = "rfft",
        norm: str | None = None,
        dtype: torch.dtype = torch.cfloat,
    ):
        """
        :param n_time_steps: number of time to produce steps, see torch.fft.irfft for details
        :param implementation: implementation of IRFFT, "rfft" uses torch.fft.irfft, "matmul" uses matrix multiplication
        :param norm: norm of IRFFT, see torch.fft.irfft for details
        :param dtype: output data type, for IRFFT should be real data type
        """
        super(IRFFTModule, self).__init__()

        self.n_time_steps = n_time_steps
        self.implementation = implementation
        self.norm = norm
        self.dtype = dtype

        self.requires_grad_(False)

        if self.implementation == "matmul":
            self.register_buffer("ifft_matrix", self.build_ifft_matrix())

    def build_ifft_matrix(self) -> Tensor:
        """Precompute FFT matrix for matmul implementation."""
        return torch.fft.ifft2(torch.eye(self.n_time_steps, dtype=self.dtype))

    def forward(self, inputs: Tensor) -> Tensor:
        if self.implementation == "rfft":
            outputs = torch.fft.irfft(inputs, n=self.n_time_steps, norm=self.norm, dim=1)
        else:  # implementation == "matmul"
            outputs = torch.matmul(self.ifft_matrix, inputs.to(self.ifft_matrix.dtype))
        return outputs.to(self.dtype)
