from typing import Callable, Iterator

from torch import Tensor
from torch.nn import Module, Parameter

# callable registering parameters of neural network for given measure
# input will be entire torch.nn.Module and output is list of names (submodule combined with parameter)
# which will be measured by given measuring function
RegisterCallable = Callable[[Module], Iterator[tuple[str, Parameter]]]
# callable measuring given parameter, input is single learnable parameter and output is float or Tensor
MeasureCallable = Callable[[Parameter], float | Tensor]
# measure register is a 4-tuple of elements needed to run given measuring function over selected params, it consists of:
# 1. measure name
# 2. registering function called over all modules and parameters
# 3. measuring function called over selected parameter returning float or Tensor
# 4. flag if the measure is to be run after every epoch (run if given as True)
MeasureRegister = tuple[str, RegisterCallable, MeasureCallable, bool]


def iter_modules_and_parameters(module: Module) -> Iterator[tuple[str, Parameter]]:
    """
    Lazy iterator over all modules and parameters of given module, yields name and the parameter (Tensor),
    name is composed of module name and parameter name. For

    :example:
        >>> model = torch.nn.Sequential(torch.nn.Linear(32, 10), torch.nn.ReLU(), torch.nn.Linear(32, 10))
        >>> names_and_parameters = list(iter_modules_and_parameters(model))
        >>> [name for name, _ in names_and_parameters]  # just get the name
        ... ["0.weight", "0.bias", "2.weight", "2.bias"]
    """
    seen = set()
    for name, submodule in module.named_modules():
        for parameter_name, parameter in submodule.named_parameters():
            if id(parameter) in seen:  # skip already seen parameters
                continue
            # remember seen parameters by memory address to avoid duplicates
            # from torch containers, such as torch.nn.Sequential
            seen.add(id(parameter))
            if name:
                yield name + "." + parameter_name, parameter
            else:
                yield parameter_name, parameter  # module name is empty


def register_all_parameters(module: Module) -> Iterator[tuple[str, Parameter]]:
    """Simple register for all parameters of given module."""
    yield from iter_modules_and_parameters(module)


def register_matrix_parameters(module: Module) -> Iterator[tuple[str, Parameter]]:
    """Simple register for all matrix parameters of given module."""
    for name, parameter in iter_modules_and_parameters(module):
        if len(parameter.shape) == 2:
            yield name, parameter


def register_square_parameters(module: Module) -> Iterator[tuple[str, Parameter]]:
    """Simple register for all square-matrix parameters of given module."""
    for name, parameter in iter_modules_and_parameters(module):
        if parameter.shape[0] == parameter.shape[1]:
            yield name, parameter
