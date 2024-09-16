from functools import wraps
from typing import Any, Callable

import lightning.pytorch as pl

from .lightning import LightningMeasure
from .states import TrainingStage, TrainingStates
from .storage import AbstractMeasureStorage


class MeasureCallback(pl.Callback):
    def __init__(
        self,
        measures: list[LightningMeasure],
        storages: list[AbstractMeasureStorage],
        persistent: bool = False,
    ):
        """
        :param measures: list of measures to be called, each measure is instance of LightningMeasure
        :param storages: list of storages to store measures, each storage is instance of AbstractMeasureStorage
        :param persistent: if True, measures will be stored after each called, otherwise only after on_train_end
        """
        super().__init__()

        self.measures = measures
        self.storages = storages
        self.persistent = persistent

    def call(self, trainer: pl.Trainer, module: Any, current_stage: TrainingStage, state: TrainingStates):
        """Function executing all measures on given stage and training state."""
        for measure in self.measures:
            if measure.measure_at == current_stage:
                for name, parameter in measure.register_fn(module):
                    value = measure(parameter)
                    for storage in self.storages:
                        storage.store(trainer, module, value, state)

    @staticmethod
    def write(always: bool = False):
        """
        Write is implemented as method-decorator, which writes to all storages after method is called.
        If `always` is set to True, storages will be written even if `persistent` is set to False.
        """

        def decorator(method: Callable):
            @wraps(method)
            def wrapper(self, *args, **kwargs):
                method(self, *args, **kwargs)
                if self.persistent or always:
                    for storage in self.storages:
                        storage.write()

            return wrapper

        return decorator

    @write
    def on_train_start(self, trainer: pl.Trainer, module: Any):
        self.call(trainer, module, TrainingStage.on_train_start, TrainingStates.initial)

    @write(always=True)
    def on_train_end(self, trainer: pl.Trainer, module: Any):
        self.call(trainer, module, TrainingStage.on_train_end, TrainingStates.final)

    @write
    def on_train_epoch_start(self, trainer: pl.Trainer, module: Any):
        self.call(trainer, module, TrainingStage.on_train_epoch_start, TrainingStates.training)

    @write
    def on_train_epoch_end(self, trainer: pl.Trainer, module: Any):
        self.call(trainer, module, TrainingStage.on_train_epoch_end, TrainingStates.training)
