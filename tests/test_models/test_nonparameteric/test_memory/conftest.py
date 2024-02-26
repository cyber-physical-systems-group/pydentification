import pytest
import torch
from torch import Tensor

from pydentification.models.nonparametric.memory import ExactMemoryManager, NNDescentMemoryManager


@pytest.fixture(scope="module")
def memory() -> Tensor:
    return torch.linspace(0, 1, 101).unsqueeze(-1)  # 101 points in [0, 1] range spaced by 0.01 and shape [101, 1]


@pytest.fixture(scope="module")
def targets(memory: Tensor) -> Tensor:
    return 2 * memory  # dummy targets


@pytest.fixture(scope="module")
def exact_memory_manager(memory: Tensor, targets: Tensor):
    manager = ExactMemoryManager()
    manager.prepare(memory, targets)

    return manager


@pytest.fixture(scope="module")
def nn_descent_memory_manager(memory: Tensor, targets: Tensor):
    manager = NNDescentMemoryManager(metric="euclidean")
    manager.prepare(memory, targets)

    return manager
