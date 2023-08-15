import torch


class ResidualConnectionWrapper(torch.nn.Module):
    """Wrapper for torch module that adds residual connection to the input"""

    def __init__(self, module: torch.nn.Module):
        super().__init__()
        self.module = module

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.module(inputs)
