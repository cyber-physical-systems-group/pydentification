# Frequency-Supported Neural Networks (FSNN)

This module implements FSNN, which is an attempted architecture we created before working on this library. Its
description is available in ArXiV pre-print ([https://arxiv.org/abs/2305.06344](https://arxiv.org/abs/2305.06344)) and
the core idea is to use Fourier transform combined with linear layer, inspired by Fourier Neural Operator [1].

The model uses `orthogonal` and `feedforward` modules from this library internally.

## Example

To create simple FSNN model, run following code:

```python
import torch

from pydentification.models.fsnn import TimeFrequencyLinear


# this is for SISO system
# for MIMO system, use TimeFrequencyLinear with n_input_state_variables > 1and n_output_state_variables > 1
model = torch.nn.Sequential(
    TimeFrequencyLinear(n_input_time_steps=64, n_output_time_steps=16, n_input_state_variables=1, n_output_state_variables=1),
    torch.nn.GELU(),  # used in the paper
    TimeFrequencyLinear(n_input_time_steps=16, n_output_time_steps=16, n_input_state_variables=1, n_output_state_variables=1)
    torch.nn.GELU(),
    TimeFrequencyLinear(n_input_time_steps=16, n_output_time_steps=1, n_input_state_variables=1, n_output_state_variables=1)
)
```

## References

<a id="1">[1]</a> 
Zongyi Li and Nikola Kovachki and Kamyar Azizzadenesheli and Burigede Liu and Kaushik Bhattacharya and Andrew Stuart and Anima Anandkumar (2021). 
*Fourier Neural Operator for Parametric Partial Differential Equations*
https://arxiv.org/abs/2010.08895
