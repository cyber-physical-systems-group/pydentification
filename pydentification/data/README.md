# Data

This module contains the data loading utils for system identification. 

## Windowing

The core functionality is window generation for time series data or system measurements. Two core identification
problems are supported: simulation modelling and predictive modelling [1]. 

## Examples

Generating a windowed dataset for simulation modelling, with input length of 64 samples of excitation and output length
of 16 samples of measured system response, where the alignment of excitation and response is from the last sample.

```python
import numpy as np
from pydentification.data import generate_time_series_windows

# some random paths for the example 
inputs = np.loadtxt("inputs.csv")
outputs = np.loadtxt("outputs.csv")

windows = generate_time_series_windows(
    inputs=inputs,
    outputs=outputs,
    shift=1,
    forward_input_window_size=64,
    forward_output_window_size=16,
    forward_output_mask=64 - 16,
)
```

Loading data for predictive modelling can be done using similar method:

```python
import numpy as np
from pydentification.data import generate_time_series_windows

# just outputs are required
outputs = np.loadtxt("outputs.csv")


windows = generate_time_series_windows(
    outputs=outputs,
    shift=1,
    backward_output_window_size=16,
    forward_output_window_size=16,
)
```

## References

<a id="1">[1]</a> 
Johan Schoukens and Lennart Ljung (2019). 
*Nonlinear System Identification: A User-Oriented Roadmap.*
https://arxiv.org/abs/1902.00683
