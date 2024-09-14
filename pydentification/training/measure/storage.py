import json
from abc import ABC, abstractmethod
from pathlib import Path

import lightning.pytorch as pl
import pandas as pd

from pydentification.stubs import Print

from .lightning import Measure
from .states import TrainingStates


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


class JsonStorage(AbstractMeasureStorage):
    """
    Storage writing measures to JSON file, with following structure:
    ```
        {
            "training":
                {
                    "0": {"name": "measure", "parameter": "parameter", "value": 0.0},
                    "1": {"name": "measure", "parameter": "parameter", "value": 0.0},
                    ...
                },
            "initial": ...,    # optional
            "final": ...,  # optional
        }
    ```
    """

    def __init__(self, path: Path):
        self.path = path

        self.storage = {str(TrainingStates.training): {}}

    def store_epoch(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure):
        if measure.representation is not None:
            for key, value in measure.representation.items():
                row = {"name": measure.name, "parameter": measure.parameter_name, "key": key, "value": value}
                self.storage[str(TrainingStates.training)][trainer.current_epoch] = row
        else:
            row = {"name": measure.name, "parameter": measure.parameter_name, "value": measure.value}
            self.storage[str(TrainingStates.training)][trainer.current_epoch] = row

    def store(
        self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure, state: str = TrainingStates.training
    ):
        """
        Store measure for given state, such as "initial" or "final". By default, "training" is used, but the parameter
        should be properly passed from pl.Callback to recognize, which measures where from which training stage.
        """
        if measure.representation is not None:
            for key, value in measure.representation.items():
                row = {"name": measure.name, "parameter": measure.parameter_name, "key": key, "value": value}
                self.storage[str(state)][trainer.current_epoch] = row
        else:
            row = {"name": measure.name, "parameter": measure.parameter_name, "value": measure.value}
            self.storage[str(state)][trainer.current_epoch] = row

    def write(self):
        with self.path.open("w") as file:
            json.dump(self.storage, file)


class CSVStorage(AbstractMeasureStorage):
    """
    Storage writing measures to CSV file, with following structure:
    ```
        name,parameter,key,value,epoch,state
        measure,linear.weight,mean,0.6,0,training
        measure,linear.weight,std,0.1,0,training
        measure,linear.weight,mean,0.1,1,training
        measure,linear.weight,std,0.01,1    ,training
        measure,linear.weight,mean,0.5,None,final
        measure,linear.weight,std,0.2,None,final
    ```
    """

    def __init__(self, path: Path):
        self.path = path
        self.storage = []

    @staticmethod
    def compose_record(measure: Measure, epoch: int | None, state: TrainingStates):
        if measure.representation is not None:
            for key, value in measure.representation.items():
                return {
                    "name": measure.name,
                    "parameter": measure.parameter_name,
                    "key": key,
                    "value": value,
                    "epoch": epoch,
                    "state": str(state),
                }

        else:
            return {
                "name": measure.name,
                "parameter": measure.parameter_name,
                "value": measure.value,
                "epoch": epoch,
                "state": str(state),
            }

    def store_epoch(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure):
        record = self.compose_record(measure, trainer.current_epoch, TrainingStates.training)
        self.storage.append(record)

    def store(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        measure: Measure,
        state: TrainingStates = TrainingStates.training,
    ):
        record = self.compose_record(measure, None, state)
        self.storage.append(record)

    def write(self):
        pd.DataFrame(self.storage).to_csv(self.path, index=False)
