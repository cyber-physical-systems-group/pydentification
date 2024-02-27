import warnings
from typing import Literal

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from torch import Tensor

from .abstract import MemoryManager


class SklearnMemoryManager(MemoryManager):
    """
    Memory manager wrapping sklearn's NearestNeighbors. This memory manager can be used for relatively large datasets,
    however for very large datasets it is recommended to use approximated nearest neighbors algorithms.

    It can be used by default with k-nn mode or radius mode.
    """

    def __init__(self, k: int, algorithm: Literal["auto", "ball_tree", "kd_tree", "brute"] = "auto", **parameters):
        """
        :param k: number of nearest neighbors used to initialize the model, it can be over-written
        :param algorithm: algorithm used to compute nearest neighbors
        :param parameters: additional parameters passed to NearestNeighbors model,
                           for more details see: see: https://scikit-learn.org/stable/modules/neighbors.html#neighbors
        """
        super().__init__()

        # placeholders stored in prepare method
        self.memory: Tensor | None = None
        self.targets: tuple[Tensor, ...] | None = None
        self.neighbors: NearestNeighbors | None = None

        self.k = k
        self.algorithm = algorithm
        self.parameters = parameters

    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]) -> None:
        """
        :param memory: tensor of indexed data points to search for nearest neighbors
        :param targets: tensor of target values corresponding to the memory points, can be any number of tensors
        """
        self.memory = memory
        self.targets = targets if isinstance(targets, tuple) else (targets,)

        self.neighbors = NearestNeighbors(n_neighbors=self.k, algorithm=self.algorithm, **self.parameters)
        self.neighbors.fit(memory)

    def to(self, device: torch.device) -> None:
        """Move memory manager to given device"""
        self.memory = self.memory.to(device)
        self.targets = tuple(target.to(device) for target in self.targets)

    def query_nearest(self, points: Tensor, k: int | None = None) -> [tuple[Tensor, ...]]:
        if k != self.k:
            warnings.warn(f"Using k different from the one used to fit the model! {k} != {self.k}")

        k = k or self.k  # default k was used to fit the neighbors model

        index = self.neighbors.kneighbors(points, n_neighbors=k, return_distance=False)
        index = np.unique(np.concatenate(index).flatten())

        return self.memory[index, :], *(target[index, :] for target in self.targets)

    def query_radius(self, points: Tensor, r: float) -> [tuple[Tensor, Tensor]]:
        index = self.neighbors.radius_neighbors(points, radius=r, return_distance=False)
        index = np.unique(np.concatenate(index).flatten())

        return self.memory[index, :], *(target[index, :] for target in self.targets)

    def query(
        self, points: Tensor, *, k: int | None = None, r: float | None = None, **kwargs
    ) -> [tuple[Tensor, Tensor]]:
        if not (k is None) ^ (r is None):
            raise ValueError("Exactly one of: k and r parameter must be specified!")

        if k is not None:
            return self.query_nearest(points, k=k)

        if r is not None:
            return self.query_radius(points, r=r)