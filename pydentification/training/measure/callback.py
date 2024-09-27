from typing import Any

import lightning.pytorch as pl

from .lightning import LightningMeasure
from .monitor import MeasureMonitor
from .states import TrainingStage, TrainingStates


class MeasureCallback(pl.Callback):
    def __init__(
        self,
        measures: list[LightningMeasure],
        monitors: list[MeasureMonitor],
        persistent: bool = False,
    ):
        """
        :param measures: list of measures to be called, each measure is instance of LightningMeasure
        :param monitors: list of storages to store measures, each storage is instance of AbstractMeasureStorage
        :param persistent: if True, measures will be stored after each call, otherwise only after on_train_end
        """
        super().__init__()

        self.measures = measures
        self.monitors = monitors
        self.persistent = persistent

    def call(
        self,
        trainer: pl.Trainer,
        module: Any,
        current_stage: TrainingStage,
        state: TrainingStates,
        write: bool = False,
    ):
        """Function executing all measures on given stage and training state."""
        for measure in self.measures:
            if current_stage in measure.measure_at:
                for value in measure(module):
                    for monitor in self.monitors:
                        monitor.log(trainer, module, value, state)

        if write:
            for monitor in self.monitors:
                monitor.persist()

    def on_train_start(self, trainer: pl.Trainer, module: Any):
        self.call(trainer, module, TrainingStage.on_train_start, TrainingStates.initial, write=self.persistent)

    def on_train_end(self, trainer: pl.Trainer, module: Any):
        # always write measures at the end of training
        self.call(trainer, module, TrainingStage.on_train_end, TrainingStates.final, write=True)

    def on_train_epoch_start(self, trainer: pl.Trainer, module: Any):
        self.call(trainer, module, TrainingStage.on_train_epoch_start, TrainingStates.training, write=self.persistent)

    def on_train_epoch_end(self, trainer: pl.Trainer, module: Any):
        self.call(trainer, module, TrainingStage.on_train_epoch_end, TrainingStates.training, write=self.persistent)
