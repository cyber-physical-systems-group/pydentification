from dataclasses import dataclass
from typing import Callable, Iterator

import torch
from torch import Tensor
from torch.nn import Module, Parameter

from .register import iter_modules_and_parameters

# callable registering parameters of neural network for given measure
# input will be entire torch.nn.Module and output is list of names (submodule combined with parameter)
# which will be measured by given measuring function
RegisterCallable = Callable[[Module], Iterator[tuple[str, Parameter]]]
# callable measuring given parameter, input is single learnable parameter and output is float or Tensor
MeasureCallable = Callable[[Parameter], float | Tensor]
# callable processing measured value, input is measured value and output is processed value
# returns float or Tensor or dictionary of values, such as statistics or multiple measures (or Tensor measure)
PostProcessCallable = Callable[[float | Tensor], float | Tensor | dict[str, float | Tensor]]


@dataclass
class Measure:
    name: str
    parameter_name: str
    value: float | Tensor
    representation: float | dict[str, float] | None


class LightningMeasure:
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
        postprocess_fn: PostProcessCallable | None = None,  # default to no processing
    ):
        """
        :param name: name of the measure, will be returned for each call
        :param measure_fn: callable measuring single parameter of neural network
        :param register_fn: callable registering parameters of neural network for given measure
        :param postprocess_fn: callable processing measured value, returns processed value or dictionary of values
        """
        self.name = name

        self.measure = measure_fn
        self.register_fn = register_fn
        self.postprocess_fn = postprocess_fn

        self.register: set[str] | None = None

    @torch.no_grad()
    def __call__(self, module: Module) -> tuple[str, str, float | Tensor]:
        if not self.register:
            self.register = set([name for name, _ in self.register_fn(module)])

        for name, parameter in iter_modules_and_parameters(module):
            if name in set(self.register):
                value = self.measure(parameter)
                if self.postprocess_fn:  # add post-processed representation
                    yield Measure(self.name, name, value, self.postprocess_fn(value))

                yield Measure(self.name, name, value, None)
