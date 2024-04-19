import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset


class StepAheadModule(Module):
    """
    Mocked torch module for testing functionalities with one-step ahead training.
    It takes any input and always returns one step of zeros with the same dimension as inputs.

    The module has single parameter, so tests of training methods computing gradients can be performed.
    Gradients are zero all the time, since the output is always constant.
    """

    def __init__(self):
        super(StepAheadModule, self).__init__()

        self.parameter = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.parameter * torch.zeros_like(x[:, 0, :])


class RandDataset(Dataset):
    """
    Random torch dataset returning given number of elements each with given shape
    """

    def __init__(self, size: int, shape: tuple):
        self.size = size
        self.shape = shape

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _) -> Tensor:
        return torch.rand(self.shape)
