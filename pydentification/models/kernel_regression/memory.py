from abc import ABC, abstractmethod
from functools import wraps
from typing import Callable

from torch import Tensor


def needs_tensor_with_dims(*expected_dims: tuple[int | None]) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(tensor: Tensor, *args, **kwargs):
            if not isinstance(tensor, Tensor):
                raise TypeError("Input should be a PyTorch Tensor")

            if tensor.dim() != len(expected_dims):
                raise ValueError(f"Expected tensor with {len(expected_dims)} dimensions, got {tensor.dim()}")

            for index, dim in enumerate(expected_dims):
                if dim is not None and tensor.size(index) != dim:
                    raise ValueError(f"Expected size {dim} at dimension {index}, got {tensor.size(index)}")

            return func(tensor, *args, **kwargs)

        return wrapper

    return decorator


class MemoryManager(ABC):
    """
    Interface for memory manager for non-parametric models (such as kernel regression).

    This class is used to enable non-parametric models to be used with large datasets, since they need to store all
    training data. Memory managers can be used to apply selection or search algorithms to reduce the number of samples
    return for each prediction, for example for kernel model returning only samples near the input points in given
    prediction batch.
    """

    def __init__(self):
        ...

    @abstractmethod
    def query_nearest(self, points: Tensor, k: int) -> [tuple[Tensor, Tensor]]:
        """
        Query for K-nearest neighbors in memory given input points.

        :param points: input points for which to find nearest neighbours
        :param k: number of nearest neighbours to return
        """
        ...

    @abstractmethod
    def query_radius(self, points: Tensor, r: float) -> [tuple[Tensor, Tensor]]:
        """
        Query for all points in memory within given radius of input points.

        :param points: input points for which to find neighbours
        :param r: radius of the neighbourhood
        """
        ...
