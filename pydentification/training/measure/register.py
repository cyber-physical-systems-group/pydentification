from typing import Callable, Iterator

from torch.nn import Module, Parameter


# callable registering parameters of neural network for given measure
# input will be entire torch.nn.Module and output is list of names (submodule combined with parameter)
# which will be measured by given measuring function
RegisterCallable = Callable[[Module], Iterator[tuple[str, Parameter]]]


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
