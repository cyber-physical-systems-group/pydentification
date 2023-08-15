# Dynamical Self Attention

This module implements self-attention model for dynamical system. It is really similar to transformer (encoder or
decoder), but it does not share feedforward network weights between layers. In classical transformer, the inputs have
shape of (batch_size, seq_len, input_dim), which is also true for this module, however the dimension of input_dim is
usually much lower and sequence length can be higher. This is due to the fact, that real-world dynamical system
typically have fewer dimensions than what is expected in transformer embeddings.

### Feedforward

Feedforward network is a linear layer, which is applied using delay-line of inputs. This means that single weight
processes single time-step for each dimension of the system. For a system with 3 dimensions and 10 steps the feedforward
network would have 30 weights (+ optionally bias parameters). There are two variants, plain delay line without
activation or transformer-like two layered feedforward with activation in between.  

### Example

```python
import torch

from pydentification.models.attention import (
    DelayLineFeedforward,
    DynamicalSelfAttention,
    LinearEmbedding,
    LinearReadout,
)

model = torch.nn.Sequential(
    # make sequence shorter in first layer
    LinearEmbedding(n_input_time_steps=64, n_input_state_variables=1, n_output_time_steps=16, n_output_state_variables=1),
    # add three layers of self-attention block with delay-line feedforward and GELU 
    DynamicalSelfAttention(n_time_steps=16, n_state_variables=1, n_heads=1, skip_connection=True),
    DelayLineFeedforward(n_time_steps=16, n_state_variables=1, skip_connection=True),
    torch.nn.GELU(),
    DynamicalSelfAttention(n_time_steps=16, n_state_variables=1, n_heads=1, skip_connection=True),
    DelayLineFeedforward(n_time_steps=16, n_state_variables=1, skip_connection=True),
    torch.nn.GELU(),
    DynamicalSelfAttention(n_time_steps=16, n_state_variables=1, n_heads=1, skip_connection=True),
    DelayLineFeedforward(n_time_steps=16, n_state_variables=1, skip_connection=True),
    torch.nn.GELU(),
    DynamicalSelfAttention(n_time_steps=16, n_state_variables=1, n_heads=1, skip_connection=True),
    DelayLineFeedforward(n_time_steps=16, n_state_variables=1, skip_connection=True),
    torch.nn.GELU(),
    # final layer producing output time-step
    LinearReadout(n_input_time_steps=16, n_output_time_steps=1, n_input_state_variables=1, n_output_state_variables=1, bias=True),
)
```