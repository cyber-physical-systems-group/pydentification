import torch
from torch.nn import Module


@torch.no_grad()
def reset_parameters(module: Module) -> None:
    """
    Reset parameters of given module, if it has reset_parameters method implemented

    :param module: any torch module, including Sequential and composite modules to reset parameters

    :example:
        >>> model.apply(reset_parameters)
    """
    reset_func = getattr(module, "reset_parameters", None)

    if callable(reset_func):
        module.reset_parameters()
