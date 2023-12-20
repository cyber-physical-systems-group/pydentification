# Orthogonal

This module is a module of utils allowing usage of orthogonal transforms (such as Fourier transform) in pytorch models.
Currently, real-valued Fourier transform is implemented and complex-valued Fourier transform are implemented. 
The models contain no trainable parameters, but are useful for quickly iterating over ideas using them and benchmarking.

## Example

To create model using FFT on input, run following code:

```python
import torch

from pydentification.models.orthogonal import RFFTModule


model = torch.nn.Sequential(
    RFFTModule(n_input_time_steps=64, dtype=torch.float32),  # optimal lengths are powers of 2, due to FFT algorithm
    # casting is done from complex to float, so phase information is lost, 
    # this is not required, torch can work with complex dtypes
    torch.nn.Linear(32, 10),
    torch.nn.ReLU(),
    torch.nn.Linear(10, 1)
)
```
