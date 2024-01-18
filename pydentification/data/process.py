from typing import Iterable, Sequence

import torch
from torch import Tensor


def decay(x: Tensor, gamma: float) -> Tensor:
    """Apply exponential decay to batched time-series data."""
    n_time_steps = x.size(1)  # get number of time steps from input tensor
    factors = torch.exp(gamma * torch.arange(n_time_steps, dtype=x.dtype, device=x.device))
    return x * factors


def lerpna(x: Tensor, slope: float) -> Tensor:
    """
    Linear interpolation of NaN values in a tensor.
    NaN values are replaced by linear interpolation between points with given slope.
    """
    if not torch.isnan(x).any():
        return x.clone()  # nothing to do

    tensor = x.clone()  # work with clone tensor

    mask = torch.isnan(tensor)
    (non_nan_indices,) = torch.where(~mask)

    for index in range(1, len(non_nan_indices)):
        # find current NaN segment
        start = non_nan_indices[index - 1]
        end = non_nan_indices[index]
        segment_length = end - start - 1

        if segment_length > 0:
            start_value = tensor[start]
            end_value = start_value + slope * segment_length
            # assume [0, 1] range
            weights = torch.linspace(0, 1, steps=segment_length)
            # linearly interpolate values
            interpolated_values = torch.lerp(start_value, end_value, weights)
            tensor[start + 1 : end] = interpolated_values  # noqa: E203

    return tensor


def unbatch(batched: Iterable[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    """
    Converts batched dataset given as iterable (usually lazy iterable) to tuple of tensors

    :example:
    >>> loader: DataLoader = get_loader(batch_size=32)  # assume get_loader is implemented
    >>> x, y = unbatch(loader)
    >>> x.shape
    ... (320, 10, 1)  # (BATCH_SIZE * N_BATCHES, *DATA_SHAPE)
    """
    for batch_idx, batch in enumerate(batched):
        if batch_idx == 0:  # initialize unbatched list of first batch
            n_tensors = len(batch) if isinstance(batch, Sequence) else 1
            unbatched = [Tensor() for _ in range(n_tensors)]

        for i, tensor in enumerate(batch):
            unbatched[i] = torch.cat([unbatched[i], tensor])

    return tuple(unbatched)
