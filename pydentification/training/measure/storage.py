import warnings
from abc import ABC, abstractmethod

import lightning.pytorch as pl
from torch import Tensor

from pydentification.stubs import Print

try:
    from lovely_tensors import lovely
except ImportError:
    warnings.warn(
        "Missing optional dependency 'lovely-tensors'."
        "Run `pip install lovely-tensors` or `pip install pydentification[experiment] to use `LoggingMeasureStorage`"
    )


class AbstractMeasureStorage(ABC):
    """
    Abstract interface for handling the storage of computed measures. They can be written to file or simply logged.
    The subclasses implementing this interface should be used in combination with MeasureCallback.
    """

    @abstractmethod
    def store_on_epoch_end(self, trainer: pl.Trainer, measure_name: str, parameter_name: str, value: float | Tensor):
        """Called with single measured value for each measure at the end of each epoch."""
        ...

    @abstractmethod
    def store_on_train_end(self, trainer: pl.Trainer, measure_name: str, parameter_name: str, value: float | Tensor):
        """Called with single measured value for each measure at the end of training."""
        ...

    def write(self):
        """
        Write stored measures to file or log. This function can be empty for storage methods writing after
        each call, for example logging or updating storage file after each epoch to save memory for large measures.
        """
        ...


class LoggingMeasureStorage(AbstractMeasureStorage):
    """Simple measure storage for logging to console."""

    def __init__(self, print_fn: Print = print, prefix: str = ""):
        self.print_fn = print_fn
        self.prefix = f" for {prefix}" if prefix else ""

    def store_on_epoch_end(self, trainer: pl.Trainer, measure_name: str, parameter_name: str, value: float | Tensor):
        if isinstance(value, Tensor):
            value = lovely(value)

        epoch = trainer.current_epoch
        self.print_fn(f"Measure {measure_name} for {self.prefix} {parameter_name} at epoch {epoch}: {value}")

    def store_on_train_end(self, trainer: pl.Trainer, measure_name: str, parameter_name: str, value: float | Tensor):
        if isinstance(value, Tensor):
            value = lovely(value)

        self.print_fn(f"Measure {measure_name} for {self.prefix} {parameter_name}: {value}")


class WandbMeasureStorage(AbstractMeasureStorage):
    def store_on_epoch_end(self, trainer: pl.Trainer, measure_name: str, parameter_name: str, value: float | Tensor):
        if isinstance(value, Tensor):
            value = lovely(value)

        trainer.log(f"measure/{measure_name}/{parameter_name}", value, on_epoch=True)

    def store_on_train_end(self, trainer: pl.Trainer, measure_name: str, parameter_name: str, value: float | Tensor):
        if isinstance(value, Tensor):
            value = lovely(value)

        trainer.log(f"measure/{measure_name}/{parameter_name}", value)
