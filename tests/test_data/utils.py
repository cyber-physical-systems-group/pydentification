import torch


def tensor_batch_iterable(n_batches: int, batch_size: int, batch_shape: tuple[int, ...], n_tensors: int):
    """
    Testing util for defining batched iterable for Tensors returning given
    number of batches with given batch size all with ones tensors
    """
    for _ in range(n_batches):
        yield tuple([torch.ones((batch_size,) + batch_shape) for _ in range(n_tensors)])
