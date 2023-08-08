# Feedforward

This module contains feedforward models, which consist entirely of linear blocks and activation functions. They are the
most standard type of architecture for system identification, therefore a very useful for benchmarking.

This module implements a building block, which is `TimeSeriesLinear` class, which is linear layer, capable of processing
multi-dimensional system (or time series). To create a model using this layer, run following code:

## Example

```python
import torch

from pydentification.models.feedforward import TimeSeriesLinear


model = torch.nn.Sequential(
    TimeSeriesLinear(1, 10),
    torch.nn.ReLU(),
    TimeSeriesLinear(10, 1)
)
```
