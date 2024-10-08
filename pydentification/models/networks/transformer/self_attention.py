import torch
from torch import Tensor, nn


def generate_square_subsequent_mask(size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """
    Generates a square mask for the sequence. The masked positions are filled with float('-inf').

    Source following the PyTorch implementation, but without default values
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/transformer.html#Transformer.generate_square_subsequent_mask
    """
    inf_tensor = torch.full((size, size), float("-inf"), dtype=dtype, device=device)
    return torch.triu(inf_tensor, diagonal=1)


class DynamicalSelfAttention(nn.Module):
    """
    This module computes self-attention for dynamical systems. It is using MultiHeadAttention from PyTorch
    with parameters prepared for running on dynamical systems.

    For details see: https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html
    """

    def __init__(
        self,
        n_time_steps: int,
        n_state_variables: int,
        n_heads: int,
        return_attention_weights: bool = False,
        bias: bool = True,
        skip_connection: bool = False,
        is_causal: bool = False,
    ):
        """
        :param n_time_steps: number of input and output time steps (they must be equal for self-attention)
        :param n_state_variables: number of state variables in the system or inner representation in the model
                                  equivalent to number of features in the input and embedding dimension in transformers
        :param n_heads: number of heads in multi-head attention
        :param return_attention_weights: if True module will return attention weights
        :param bias: if True bias will be added to the attention
        :param skip_connection: if True skip connection will be added to the output
        :param is_causal: if True, causal mask will be applied to the attention
        """
        super(DynamicalSelfAttention, self).__init__()

        self.n_time_steps = n_time_steps
        self.n_state_variables = n_state_variables
        self.n_heads = n_heads

        self.return_attention_weights = return_attention_weights
        self.skip_connection = skip_connection
        self.is_causal = is_causal

        self.attention = nn.MultiheadAttention(
            embed_dim=self.n_state_variables,
            num_heads=n_heads,
            bias=bias,
            # force input to be (batch, time_steps, state_variables)
            # in docs time_steps = seq and state_variables = feature
            batch_first=True,
        )

    def forward(self, inputs: Tensor) -> Tensor:
        if self.is_causal:
            # generate upper triangular mask for causal attention
            mask = generate_square_subsequent_mask(size=self.n_time_steps, device=inputs.device, dtype=inputs.dtype)
        else:
            mask = None

        outputs, weights = self.attention(
            query=inputs, key=inputs, value=inputs, attn_mask=mask, is_causal=self.is_causal
        )

        if self.skip_connection:
            outputs = inputs + outputs

        if self.return_attention_weights:
            return outputs, weights

        return outputs
