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
    def store(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure, state: TrainingStates):
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

    def store(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure, state: TrainingStates):
        value = measure.representation if measure.representation is not None else measure.value

        if state is not TrainingStates.training:
            epoch = trainer.current_epoch
            self.print_fn(
                f"Measure {measure.name} for {self.prefix} {measure.parameter_name} at epoch {epoch}: {value}"
            )

        else:
            self.print_fn(f"Measure {measure.name} for {self.prefix} {measure.parameter_name}: {value}")


class WandbMeasureStorage(AbstractMeasureStorage):
    """
    Measure storage for logging to W&B interface. It can log any data-type, which is supported by W&B, see:
    https://docs.wandb.ai/ref/python/log and https://docs.wandb.ai/ref/python/data-types/
    """

    # lightning training states, for which self.log can be used
    allowed_states = frozenset({TrainingStates.training, TrainingStates.initial})

    @staticmethod
    def store_with_wandb(module: pl.LightningModule, measure: Measure):
        if measure.representation is not None:
            for key, value in measure.representation.items():
                module.log(f"measure/{measure.name}/{measure.parameter_name}/{key}", value, on_epoch=True)
        else:
            module.log(f"measure/{measure.name}/{measure.parameter_name}", measure.value, on_epoch=True)

    def store(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure, state: TrainingStates):
        if state in self.allowed_states:
            self.store_with_wandb(module, measure)


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

    def __init__(self, path: str | Path):
        self.path = path if isinstance(path, Path) else Path(path)
        self.storage = {str(TrainingStates.training): {}}

    def store_item(
        self, state: TrainingStates, epoch: int, name: str, parameter_name: str, value: float, key: str | None = None
    ):
        row = {"name": name, "parameter": parameter_name, "value": value}

        if key is not None:
            row["key"] = key

        if state == TrainingStates.training:
            self.storage[str(state)][epoch] = row
        else:
            self.storage[str(state)] = row

    def store(self, trainer: pl.Trainer, module: pl.LightningModule, measure: Measure, state: TrainingStates):
        """
        Store measure for given state, such as "initial" or "final". By default, "training" is used, but the parameter
        should be properly passed from pl.Callback to recognize, which measures where from which training stage.
        """
        if measure.representation is not None:
            for key, value in measure.representation.items():
                # store measure with a key, mean, std, etc.
                self.store_item(state, trainer.current_epoch, measure.name, measure.parameter_name, value, key)
        else:
            # store without a key for single value
            self.store_item(state, trainer.current_epoch, measure.name, measure.parameter_name, measure.value)

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

    def __init__(self, path: str | Path):
        self.path = path if isinstance(path, Path) else Path(path)
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

    def store(
        self,
        trainer: pl.Trainer,
        module: pl.LightningModule,
        measure: Measure,
        state: TrainingStates = TrainingStates.training,
    ):
        if state is TrainingStates.training:
            record = self.compose_record(measure, trainer.current_epoch, state)
            self.storage.append(record)
        else:
            record = self.compose_record(measure, None, state)
            self.storage.append(record)

    def write(self):
        pd.DataFrame(self.storage).to_csv(self.path, index=False)
