from abc import abstractmethod
from bisect import bisect_right
from collections import Counter
from typing import Any, Literal, Sequence

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
            print(f"{self.__class__.__name__}: initial length = {trainer.datamodule.n_forward_time_steps}")

        self.base_length = trainer.datamodule.n_forward_time_steps

    def on_train_epoch_start(self, trainer: pl.Trainer, _: Any) -> None:
        if self.base_length is None:
            raise RuntimeError("{self.__class__.__name__}: base_length is None!")

        if trainer.current_epoch % self.step_size == 0:
            trainer.datamodule.n_forward_time_steps = self._get_closed_form_ar_length(trainer.current_epoch)

            if self.verbose:
                print(
                    f"{self.__class__.__name__}: new length = {trainer.datamodule.n_forward_time_steps}"
                    f" at epoch {trainer.current_epoch}"
                )


class MultiStepAutoRegressionLengthScheduler(AbstractAutoRegressionLengthScheduler):
    """
    Increases the length of auto-regression by gamma once the number of epoch reaches one of the milestones.
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
            print(f"{self.__class__.__name__}: initial length = {trainer.datamodule.n_forward_time_steps}")

        self.base_length = trainer.datamodule.n_forward_time_steps

    def on_train_epoch_start(self, trainer: pl.Trainer, _: Any) -> None:
        if self.base_length is None:
            raise RuntimeError("MultiStepAutoRegressionLengthScheduler: base_length is None")

        trainer.datamodule.n_forward_time_steps = self._get_closed_form_ar_length(trainer.current_epoch)

        if self.verbose:
            print(
                f"{self.__class__.__name__}: new length = {trainer.datamodule.n_forward_time_steps}"
                f" at epoch {trainer.current_epoch} with milestones {list(self.milestones.keys())}"
            )


class IncreaseAutoRegressionLengthOnPlateau(AbstractAutoRegressionLengthScheduler):
    """
    Increases the length of auto-regression by factor once the monitored quantity stops improving.
    Works as ReduceLROnPlateau scheduler, but increasing the length (given as int!) instead of decaying learning rate.

    :note: this callback changes the length after validation,
           at the end of epoch, unlike others, which do it on the start of epoch

    Source reference: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html
    """

    def __init__(
        self,
        monitor: str,
        patience: int,
        factor: int,
        threshold: float = 1e-4,
        threshold_mode: Literal["abs", "rel"] = "rel",
        max_length: int | None = None,
        verbose: bool = False,
    ):
        """
        :param monitor: quantity to be monitored given as key from callback_metrics dictionary of pl.Trainer
        :param patience: number of epochs with no improvement after which auto-regression length will be increased
        :param factor: factor by which to increase auto-regression length. new_length = old_length * factor
        :param threshold: threshold for measuring the new optimum, to only focus on significant changes
        :param threshold_mode: one of {"rel", "abs"}, defaults to "rel"
        :param max_length: maximum auto-regression length, defaults to None
        :param verbose: if True, prints the auto-regression length when it is changed
        """
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.factor = factor

        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.max_length = max_length
        self.verbose = verbose

        self.best = float("inf")
        self.num_bad_epochs = 0

    def on_train_start(self, trainer: pl.Trainer, _: Any) -> None:
        if self.verbose:
            print(f"{self.__class__.__name__}: initial length = {trainer.datamodule.n_forward_time_steps}")

    def is_better(self, current: float, best: float) -> bool:
        if self.threshold_mode == "rel":
            return current < best * (float(1) - self.threshold)

        else:  # self.threshold_mode == "abs":
            return current < best - self.threshold

    def on_validation_epoch_end(self, trainer: pl.Trainer, _: Any) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            raise RuntimeError(f"{self.__class__.__name__}: metric {self.monitor} not found in callback_metrics!")

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            new_length = trainer.datamodule.n_forward_time_steps * self.factor

            if new_length > self.max_length:
                if self.verbose:
                    print(f"{self.__class__.__name__}: maximum length reached, not increasing")
                return  # exit function is new length is greater than maximum length

            trainer.datamodule.n_forward_time_steps = new_length
            self.num_bad_epochs = 0

            if self.verbose:
                print(
                    f"{self.__class__.__name__}: new length = {trainer.datamodule.n_forward_time_steps}"
                    f" at epoch {trainer.current_epoch}"
                )
