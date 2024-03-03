from abc import ABC, abstractmethod

from torch import Tensor


class MemoryTransformation(ABC):
    """
    Interface for memory transformations, which can be used to modify memory and query points before they are used in
    memory manager and after, to return them to kernel regression model.

    This can be used to decrease dimension of memory and query points, normalize the data for the nearest neighbour
    search or to make data points further in time (using delay-line) further apart in memory space,
    since they are less important for prediction.
    """

    @abstractmethod
    def before_prepare(self, memory: Tensor) -> Tensor:
        ...

    @abstractmethod
    def before_query(self, query: Tensor) -> Tensor:
        ...

    @abstractmethod
    def after_query(self, memory: Tensor) -> Tensor:
        ...


class TruncateDelayLine(MemoryTransformation):
    """
    Truncate delay line to last `target_dim` elements in order to make the nearest neighbour search faster and allow
    kernel model to use only the most recent data for prediction, therefore decreasing the effective bandwidth.

    This memory transformation changes points both for nearest neighbour search and kernel regression.
    """

    def __init__(self, target_dim: int):
        self.target_dim = target_dim

    def before_prepare(self, memory: Tensor) -> Tensor:
        return memory[:, -self.target_dim :]  # noqa E203

    def before_query(self, query: Tensor) -> Tensor:
        return query[:, -self.target_dim :]  # noqa E203

    def after_query(self, memory: Tensor) -> Tensor:
        return memory  # memory is modified during before_prepare call
