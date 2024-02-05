from abc import ABC, abstractmethod

import torch
from torch import Tensor


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
    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]) -> None:
        """Prepare memory manager for use, for example build index for nearest neighbors search"""
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

    @abstractmethod
    def to(self, device: torch.device) -> None:
        """Move memory manager to given device"""
        ...

    @abstractmethod
    def __call__(self, points: Tensor, *args, **kwargs) -> [tuple[Tensor, Tensor]]:
        """Default call method, can be different for different memory managers"""
        ...
