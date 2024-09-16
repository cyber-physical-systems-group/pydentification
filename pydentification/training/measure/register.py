from typing import Iterator, Type

from torch.nn import Module, Parameter


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
        if len(parameter.shape) == 2 and parameter.shape[0] == parameter.shape[1]:
            yield name, parameter


class RegisterInstances:
    """
    Callable class returning all modules, which are instances of given class.
    This can be used to register all Linear, LSTM or other specific torch.nn.Module instances.
    """

    def __init__(self, instance: Type[Module]):
        self.instance = instance

    def __call__(self, module: Module) -> Iterator[tuple[str, Parameter]]:
        for name, submodule in module.named_modules():
            if isinstance(submodule, self.instance):
                yield name, module
