from abc import ABC, abstractmethod

import lightning.pytorch as pl

from pydentification.stubs import Print

from .lightning import Measure


class AbstractMeasureStorage(ABC):
    """
    Abstract interface for handling the storage of computed measures. They can be written to file or simply logged.
    The subclasses implementing this interface should be used in combination with MeasureCallback.
    """

    @abstractmethod
    def store_epoch(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure):
        """Called with single measured value for each measure at the end of each epoch."""
        ...

    @abstractmethod
    def store(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure):
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

    def store_epoch(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure):
        value = measure.representation if measure.representation is not None else measure.value
        epoch = trainer.current_epoch

        self.print_fn(f"Measure {measure.name} for {self.prefix} {measure.parameter_name} at epoch {epoch}: {value}")

    def store(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure):
        value = measure.representation if measure.representation is not None else measure.value
        self.print_fn(f"Measure {measure.name} for {self.prefix} {measure.parameter_name}: {value}")


class WandbMeasureStorage(AbstractMeasureStorage):
    """
    Measure storage for logging to W&B interface. It can log any data-type, which is supported by W&B, see:
    https://docs.wandb.ai/ref/python/log and https://docs.wandb.ai/ref/python/data-types/
    """

    @staticmethod
    def store_with_wandb(module: pl.LightningModule, measure: Measure, on_epoch: bool):
        if measure.representation is not None:
            for key, value in measure.representation.items():
                module.log(f"measure/{measure.name}/{measure.parameter_name}/{key}", value, on_epoch=on_epoch)
        else:
            module.log(f"measure/{measure.name}/{measure.parameter_name}", measure.value, on_epoch=on_epoch)

    def store_epoch(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure):
        self.store_with_wandb(module, measure, on_epoch=True)

    def store(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure):
        self.store_with_wandb(module, measure, on_epoch=False)
