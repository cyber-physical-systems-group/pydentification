import torch
from torch import Tensor

from .abstract import MemoryManager

try:
    from pynndescent import NNDescent
except ImportError as ex:
    message = (
        "Missing optional dependency, to install all optionals from experiment module run:\n"
        "`pip install -r pydentification/models/kernel_regression/extra-requirements.txt`"
    )

    raise ImportError(message) from ex


class NNDescentMemoryManager(MemoryManager):
    def __init__(self, **parameters):
        """
        :param parameters: parameters for NNDescent algorithm,
                           see: https://pynndescent.readthedocs.io/en/latest/api.html
        """
        super().__init__()

        # placeholders stored in prepare method
        self.memory: Tensor | None = None
        self.targets: tuple[Tensor, ...] | None = None

        self.index = None  # build deferred until first query, it takes significant amount of time
        self.parameters = parameters

    def prepare(self, memory: Tensor, targets: Tensor | tuple[Tensor, ...]) -> None:
        """
        Build index for nearest neighbors search over memory using `NNDescent`

        :param memory: tensor of indexed data points to search for nearest neighbors
        :param targets: tensor of target values corresponding to the memory points, can be any number of tensors
        """
        self.memory = memory
        self.targets = targets if isinstance(targets, tuple) else (targets,)  # store targets as tuple

        self.index = NNDescent(self.memory, **self.parameters)
        self.index.prepare()

    @property
    def neighbor_graph(self) -> tuple[Tensor, Tensor]:
        """Returns the neighbor graph of the memory points"""
        if self.index is None:
            raise RuntimeError("Index is not built, call prepare method first!")

        return self.index.neighbor_graph

    def query_nearest(self, points: Tensor, k: int, epsilon: float = 0.1) -> [tuple[Tensor, ...]]:
        """
        Query for K-nearest neighbors in memory given input points.

        :param points: input points for which to find nearest neighbours
        :param k: number of nearest neighbours to return
        :param epsilon: search parameter for NNDescent, see: https://pynndescent.readthedocs.io/en/latest/api.html
        """
        return_device = self.memory.device

        if points.device != self.memory.device:
            return_device = points.device  # remember device where points came from
            points = points.to(self.memory.device)  # move points to memory device, since NNDescent only supports CPU

        if self.index is None:
            raise RuntimeError("Index is not built, call prepare method first!")

        indexed, _ = self.index.query(points, k=k, epsilon=epsilon)
        # memory manager returns flat memory for all query points
        # duplicates are removed and the dimensionality is reduced to 1
        indexed = torch.unique(torch.from_numpy(indexed.flatten()))
        # return found nearest points from memory and collect from all target tensors corresponding to them
        memory = self.memory[indexed, :].to(return_device)  # cast back to device where points came from
        targets = tuple(target[indexed, :].to(return_device) for target in self.targets)

        return memory, *targets

    def query_radius(self, points: Tensor, r: float) -> [tuple[Tensor, Tensor]]:
        raise NotImplementedError("Radius query is not implemented for NNDescentMemoryManager!")

    def to(self, device: torch.device) -> None:
        if device != torch.device("cpu"):
            raise ValueError("NNDescentMemoryManager only supports CPU device!")

    def __call__(
        self, points: Tensor, k: int | None = None, epsilon: float | None = None, *args, **kwargs
    ) -> [tuple[Tensor, Tensor]]:
        """
        Default call method for NNDescentMemoryManager is using k-nearest neighbors search
        Using __call__ should be done in parameterized setting, where different managers can appear
        """
        if k is None:
            raise ValueError("K-nearest neighbors k parameter must be specified!")

        return self.query_nearest(points, k=k, epsilon=epsilon)
