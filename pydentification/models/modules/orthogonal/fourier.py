import torch
from torch import Tensor
from torch.nn import Module


class RFFTModule(Module):
    """
    Module computes Fourier Transform for real input.

    It processes tensors with shape (BATCH, TIME_STEPS, SYSTEM_DIMENSIONS). For processing inputs of system with single
    state dimension pass unsqueezed Tensor with shape: (BATCH, TIME_STEPS, 1).
    """

    def __init__(self, n_time_steps: int | None = None, norm: str | None = None, dtype: torch.dtype = torch.cfloat):
        """
        :param n_time_steps: number of time to produce steps, see torch.fft.rfft for details
        :param norm: norm of RFFT, see torch.fft.rfft for details
        :param dtype: output data type, defaults to cfloat and should be complex
                      otherwise will cause loss of information after the RFFT transform
        """
        super(RFFTModule, self).__init__()

        self.n_input_time_steps = n_time_steps
        self.norm = norm
        self.dtype = dtype

        self.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = torch.fft.rfft(inputs, n=self.n_input_time_steps, norm=self.norm, dim=1)
        return outputs.to(self.dtype)


class CFFTModule(Module):
    """
    Module computes Fourier Transform for any input.

    It processes tensors with shape (BATCH, TIME_STEPS, SYSTEM_DIMENSIONS). For processing inputs of system with single
    state dimension pass unsqueezed Tensor with shape: (BATCH, TIME_STEPS, 1).
    """

    def __init__(self, n_time_steps: int | None = None, norm: str | None = None, dtype: torch.dtype = torch.cfloat):
        """
        :param n_time_steps: number of time to produce steps, see torch.fft.fft2 for details
        :param norm: norm of FFT, see torch.fft.fft2 for details
        :param dtype: output data type, defaults to cfloat and should be complex
                      otherwise will cause loss of information after the RFFT transform
        """
        super(CFFTModule, self).__init__()

        self.n_time_steps = n_time_steps
        self.norm = norm
        self.dtype = dtype

        self.requires_grad_(False)  # non-trainable layer

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = torch.fft.fft2(inputs, norm=self.norm, n=self.n_time_steps, dim=1)
        return outputs.to(self.dtype)


class IRFFTModule(Module):
    """
    Module computes Inverse Fourier Transform converting to real output.

    It processes tensors with shape (BATCH, FREQUENCY_MODELS, SYSTEM_DIMENSIONS). For processing inputs of system with
    single state dimension pass unsqueezed Tensor with shape: (BATCH, FREQUENCY_MODELS, 1).
    """

    def __init__(self, n_time_steps: int | None = None, norm: str | None = None, dtype: torch.dtype = torch.float32):
        """
        :param n_time_steps: number of time to produce steps, see torch.fft.irfft for details
        :param norm: norm of IRFFT, see torch.fft.irfft for details
        :param dtype: output data type, for IRFFT should be real data type
        """
        super(IRFFTModule, self).__init__()

        self.n_time_steps = n_time_steps
        self.norm = norm
        self.dtype = dtype

        self.requires_grad_(False)

    def forward(self, inputs: Tensor) -> Tensor:
        outputs = torch.fft.irfft(inputs, n=self.n_time_steps, norm=self.norm, dim=1)
        return outputs.to(self.dtype)
