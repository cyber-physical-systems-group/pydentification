import torch
from torch import Tensor
from torch.nn import Module


class PositionalEncoding(Module):
    """
    Module adding positional encoding to the input time series. This encoding is based on the original transformer
    paper (Attention is all you need) with variables adjusted to the dynamical systems' data.
    """

    def __init__(self, n_time_steps: int, n_state_variables: int):
        """
        :param n_time_steps: number of time steps in the input signal
        :param n_state_variables: number of state variables in the input signal
        """
        super(PositionalEncoding, self).__init__()

        self.n_time_steps = n_time_steps
        self.n_state_variables = n_state_variables

        # precompute the positional encodings and register them as a buffer
        self.register_buffer("encodings", self.precompute())
        self.requires_grad_(False)

    def precompute(self):
        position = torch.arange(0, self.n_time_steps, dtype=torch.float).unsqueeze(1)
        even_index = torch.arange(0, self.n_state_variables, 2).float()
        log_term = -torch.log(torch.tensor(10000.0)) / self.n_state_variables
        div_term = torch.exp(even_index * log_term)

        encodings = torch.zeros(self.n_time_steps, self.n_state_variables)
        encodings[:, 0::2] = torch.sin(position * div_term)  # sine-wave for even indices
        encodings[:, 1::2] = torch.cos(position * div_term)  # cosine-wave for odd indices

        return encodings

    def forward(self, inputs: Tensor) -> Tensor:
        _, n_input_time_steps, _ = inputs.shape
        return inputs + self.encodings[:n_input_time_steps].unsqueeze(0).expand(inputs.shape[0], -1, -1)
