from typing import Literal

import torch
from torch.nn import Module

# see: https://pytorch.org/docs/stable/generated/torch.linalg.norm.html#torch.linalg.norm
SupportedNorms = Literal["fro", "nuc", "inf", "nuc", 1, 2, 0, -1, 2]


@torch.no_grad()
def parameter_norm(
    module: Module, weight_ord: SupportedNorms | None = None, bias_ord: SupportedNorms | None = None
) -> dict[str, float]:
    """
    Compute norm of parameters of given neural-network module.

    :param module: neural-network module, norms will be computed for all elements given by named_parameters
    :param weight_ord: order of the norm, see torch.linalg.norm for details
    :param bias_ord: order of the norm for bias parameters, if None, ord is used
                     this can be used, when different norm is required for bias parameters (vectors vs matrix norm)
    """
    if bias_ord is None:
        bias_ord = weight_ord

    response = {}
    for name, param in module.named_parameters():
        if "bias" in name:
            response[name] = torch.linalg.norm(param, ord=bias_ord).item()
        else:  # weight
            response[name] = torch.linalg.norm(param, ord=weight_ord).item()

    return response
