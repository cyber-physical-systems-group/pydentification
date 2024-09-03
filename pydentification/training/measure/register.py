from typing import Callable, Iterator

from torch import Tensor
from torch.nn import Module, Parameter

# callable registering parameters of neural network for given measure
# input will be entire torch.nn.Module and output is list of names (submodule combined with parameter)
# which will be measured by given measuring function
RegisterCallable = Callable[[Module], Iterator[tuple[str, Parameter]]]
# callable measuring given parameter, input is single learnable parameter and output is float or Tensor
MeasureCallable = Callable[[Parameter], float | Tensor]


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


class MeasureRegister:
    """
    Measure register is a callable object for measuring model or layer with given measure function. It can be used
    inside pl.Callback or as standalone function. `measure_fn` and `register_fn` need to follow interfaces defined
    by MeasureCallable and RegisterCallable.

    Return type for each call is tuple of
    * Measure name (given in `__init__`)
    * Parameter name
    * Measured value as float or Tensor.
    """

    def __init__(
        self,
        name: str,
        measure_fn: MeasureCallable,
        register_fn: RegisterCallable = iter_modules_and_parameters,  # default to registering all parameters
        on_train_start: bool = False,
        on_train_end: bool = True,
        on_train_epoch_start: bool = False,
        on_train_epoch_end: bool = False,
    ):
        """
        :param name: name of the measure, will be returned for each call
        :param measure_fn: callable measuring single parameter of neural network
        :param register_fn: callable registering parameters of neural network for given measure
        :param on_train_start: whether to measure at the start of training
        :param on_train_end: whether to measure at the end of training
        :param on_train_epoch_start: whether to measure at the start of each epoch
        :param on_train_epoch_end: whether to measure at the end of each epoch
        """
        self.name = name
        self.measure = measure_fn
        self.register_fn = register_fn

        self.on_train_start = on_train_start
        self.on_train_end = on_train_end
        self.on_train_epoch_start = on_train_epoch_start
        self.on_train_epoch_end = on_train_epoch_end

        self.register: set[str] | None = None

    def __call__(self, module: Module) -> tuple[str, str, float | Tensor]:
        if not self.register:
            self.register = set([name for name, _ in self.register_fn(module)])

        for name, parameter in iter_modules_and_parameters(module):
            if name in set(self.register):
                yield self.name, name, self.measure(parameter)
