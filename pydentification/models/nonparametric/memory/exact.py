from typing import Literal

import torch
from torch import Tensor

from .abstract import MemoryManager


class ExactMemoryManager(MemoryManager):
    """
    Exact memory manager that stores all data points in memory and returns them for each query.

    This memory manager is not suitable for large datasets, but can be used for testing and debugging and small
    datasets or low-dimensional systems, where the memory is not too large.

    It can be used by default with k-nn mode or radius mode.
    """

    def __init__(self, default_call: Literal["nearest", "radius"] = "nearest"):
        """
        :param default_call: default method to call when memory manager is called as a function can be called with
                             "nearest" or "radius" mode, fetching closest points or points within given radius
        """
        super().__init__()

        self.default_call = default_call

        # placeholders stored in prepare method
        self.memory: Tensor | None = None
        self.targets: tuple[Tensor, ...] | None = None

    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]) -> None:
        """
        :param memory: tensor of indexed data points to search for nearest neighbors
        :param targets: tensor of target values corresponding to the memory points, can be any number of tensors
        """
        self.memory = memory  # store entire tensors in memory for exact search
        self.targets = targets if isinstance(targets, tuple) else (targets,)  # store targets as tuple

    def query_nearest(self, points: Tensor, k: int) -> [tuple[Tensor, ...]]:
        """
        Query for K-nearest neighbors in memory given input points.

        :param points: input points for which to find nearest neighbours
        :param k: number of nearest neighbours to return
        """
        distances = torch.cdist(points, self.memory)
        # get k nearest points for each query point
        _, index = torch.topk(distances, k=k, largest=False, dim=1)
        # flatten index tensor and remove duplicates, order of points to return does not matter
        index = torch.unique(torch.flatten(index))
        # return found nearest points from memory and collect from all target tensors corresponding to them
        return self.memory[index, :], *(target[index, :] for target in self.targets)

    def query_radius(self, points: Tensor, r: float) -> [tuple[Tensor, Tensor]]:
        """
        Query for all points in memory within given radius of input points.

        :param points: input points for which to find neighbours
        :param r: radius of the neighbourhood
        """
        distances = torch.cdist(points, self.memory)
        # check if any of the points in memory is within given radius of any query point
        mask = (distances <= r).any(axis=0)
        (index,) = torch.where(mask)

        return self.memory[index, :], *(target[index, :] for target in self.targets)

    def to(self, device: torch.device) -> None:
        """Move memory manager to given device"""
        self.memory = self.memory.to(device)
        self.targets = tuple(target.to(device) for target in self.targets)

    def __call__(self, points: Tensor, k: int | None = None, r: float | None = None) -> [tuple[Tensor, Tensor]]:
        """
        Default call method for ExactMemoryManager is controlled by `__init__`
        Using __call__ should be done in parameterized setting, where different managers can appear
        """
        if self.default_call == "nearest":
            if k is None:
                raise ValueError("K-nearest neighbors k parameter must be specified!")
            return self.query_nearest(points, k=k)
        elif self.default_call == "radius":
            if r is None:
                raise ValueError("Radius r parameter must be specified!")
            return self.query_radius(points, r=r)

        raise ValueError(f"Unknown default call method: {self.default_call}!")
