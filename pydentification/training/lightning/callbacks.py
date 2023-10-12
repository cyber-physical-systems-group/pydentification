from abc import abstractmethod
from bisect import bisect_right
from collections import Counter
from typing import Any, Sequence

import lightning.pytorch as pl


class AbstractAutoRegressionLengthScheduler(pl.Callback):
    """
    Interface for auto-regression length scheduler

    The scheduler is used to control the length of auto-regression in prediction training, following the idea that
    shorter auto-regression is easier to learn and longer auto-regression is harder to learn, so model should see more
    of shorter auto-regression early in the training.

    :warning: the callback is meant to be used with `LightningPredictionTrainingModule` and `PredictionDataModule`.
    """

    def __init__(self):
        ...

    @abstractmethod
    def on_train_epoch_start(self, trainer: pl.Trainer, _: Any) -> None:
        ...


class StepAutoRegressionLengthScheduler(AbstractAutoRegressionLengthScheduler):
    """
    Increases the length of auto-regression by gamma every step_size epochs.
    Works as StepLR scheduler, but increasing the length (given as int!) instead of decaying.

    Source reference: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#StepLR
    """

    def __init__(self, step_size: int, gamma: int, verbose: bool = False):
        """
        :param step_size: period of auto-regression length increase
        :param gamma: multiplicative factor of auto-regression length increase, defaults to 2
                      :warning: must be int and >= 1
        :param verbose: if True, prints the auto-regression length when it is changed
        """
        super().__init__()

        self.step_size = step_size
        self.gamma = gamma
        self.verbose = verbose

        self.base_length: int | None = None

    def _get_closed_form_ar_length(self, epoch: int) -> int:
        return self.base_length * self.gamma ** (epoch // self.step_size)

    def on_train_start(self, trainer: pl.Trainer, _: Any) -> None:
        if self.verbose:
            print(f"StepAutoRegressionLengthScheduler: initial length = {trainer.datamodule.n_forward_time_steps}")

        self.base_length = trainer.datamodule.n_forward_time_steps

    def on_train_epoch_start(self, trainer: pl.Trainer, _: Any) -> None:
        if self.base_length is None:
            raise RuntimeError("StepAutoRegressionLengthScheduler: base_length is None")

        if trainer.current_epoch % self.step_size == 0:
            trainer.datamodule.n_forward_time_steps = self._get_closed_form_ar_length(trainer.current_epoch)

            if self.verbose:
                print(
                    f"StepAutoRegressionLengthScheduler: new length = {trainer.datamodule.n_forward_time_steps}"
                    f" at epoch {trainer.current_epoch}"
                )


class MultiStepAutoRegressionLengthScheduler(AbstractAutoRegressionLengthScheduler):
    """
    Increases the length of auto-regression by gamma gamma once the number of epoch reaches one of the milestones.
    Works as MultiStepLR scheduler, but increasing the length (given as int!) instead of decaying.

    Source reference: https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#MultiStepLR
    """

    def __init__(self, milestones: Sequence[int], gamma: int = 2, verbose: bool = False):
        """
        :param milestones: sequence of epoch indices, must be increasing.
        :param gamma: multiplicative factor of auto-regression length increase, defaults to 2
                      :warning: must be int and >= 1
        :param verbose: if True, prints the auto-regression length when it is changed
        """
        super().__init__()

        self.milestones = Counter(milestones)
        self.gamma = gamma
        self.verbose = verbose

        self.base_length: int | None = None

    def _get_closed_form_ar_length(self, epoch: int) -> int:
        milestones = sorted(self.milestones.elements())
        return self.base_length * self.gamma ** bisect_right(milestones, epoch)

    def on_train_start(self, trainer: pl.Trainer, _: Any) -> None:
        if self.verbose:
            print(f"MultiStepAutoRegressionLengthScheduler: initial length = {trainer.datamodule.n_forward_time_steps}")

        self.base_length = trainer.datamodule.n_forward_time_steps

    def on_train_epoch_start(self, trainer: pl.Trainer, _: Any) -> None:
        if self.base_length is None:
            raise RuntimeError("MultiStepAutoRegressionLengthScheduler: base_length is None")

        trainer.datamodule.n_forward_time_steps = self._get_closed_form_ar_length(trainer.current_epoch)

        if self.verbose:
            print(
                f"MultiStepAutoRegressionLengthScheduler: new length = {trainer.datamodule.n_forward_time_steps}"
                f" at epoch {trainer.current_epoch} with milestones {list(self.milestones.keys())}"
            )
