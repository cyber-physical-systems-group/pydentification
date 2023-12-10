from abc import ABC, abstractmethod
from functools import wraps
from typing import Any, Callable

import torch
from torch import Tensor

try:
    from pynndescent import NNDescent
except ImportError as ex:
    message = (
        "Missing optional dependency, to install all optionals from experiment module run:\n"
        "`pip install -r pydentification/models/kernel_regression/extra-requirements.txt`"
    )

    raise ImportError(message) from ex


def needs_tensor_with_dims(*expected_dims: tuple[int | None, ...]) -> Callable:
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self: Any, tensor: Tensor, *args, **kwargs):
            if not isinstance(tensor, Tensor):
                raise TypeError(f"Input should be a PyTorch Tensor, got {type(tensor)}")

            if tensor.dim() != len(expected_dims):
                raise ValueError(f"Expected tensor with {len(expected_dims)} dimensions, got {tensor.dim()}")

            for index, dim in enumerate(expected_dims):
                if dim is not None and tensor.size(index) != dim:
                    raise ValueError(f"Expected size {dim} at dimension {index}, got {tensor.size(index)}")

            return func(self, tensor, *args, **kwargs)

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


class NNDescentMemoryManager(MemoryManager):
    def __init__(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...], **nn_descent_params):
        """
        :param memory: tensor of indexed data points to search for nearest neighbors
        :param targets: tensor of target values corresponding to the memory points, can be any number of tensors
        :param nn_descent_params: parameters for NNDescent algorithm,
                                  see: https://pynndescent.readthedocs.io/en/latest/api.html
        """
        super().__init__()

        self.memory = memory
        self.targets = targets if isinstance(targets, tuple) else (targets,)  # store targets as tuple

        self.index = None  # build deferred until first query, it takes significant amount of time
        self.nn_descent_params = nn_descent_params

    def _build_index(self) -> None:
        """Build index for nearest neighbors search over memory using `NNDescent`"""
        self.index = NNDescent(self.memory, **self.nn_descent_params)
        self.index.prepare()

    @property
    def neighbor_graph(self) -> tuple[Tensor, Tensor]:
        """Returns the neighbor graph of the memory points"""
        if self.index is None:
            self._build_index()

        return self.index.neighbor_graph

    def query_nearest(self, points: Tensor, k: int, epsilon: float = 0.1) -> [tuple[Tensor, ...]]:
        """
        Query for K-nearest neighbors in memory given input points.

        :param points: input points for which to find nearest neighbours
        :param k: number of nearest neighbours to return
        :param epsilon: search parameter for NNDescent, see: https://pynndescent.readthedocs.io/en/latest/api.html
        """
        if self.index is None:
            self._build_index()

        indexed, _ = self.index.query(points, k=k, epsilon=epsilon)
        # memory manager returns flat memory for all query points
        # duplicates are removed and the dimensionality is reduced to 1
        indexed = torch.unique(torch.from_numpy(indexed.flatten()))
        # return found nearest points from memory and collect from all target tensors corresponding to them
        return self.memory[indexed, :], *(target[indexed, :] for target in self.targets)

    def query_radius(self, points: Tensor, r: float) -> [tuple[Tensor, Tensor]]:
        raise NotImplementedError("Radius query is not implemented for NNDescentMemoryManager!")
