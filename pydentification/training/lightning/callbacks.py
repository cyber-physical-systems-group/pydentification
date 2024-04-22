from bisect import bisect_right
from collections import Counter
from typing import Any, Literal, Sequence

import lightning.pytorch as pl


class StepAutoRegressionLengthScheduler(pl.Callback):
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


class MultiStepAutoRegressionLengthScheduler(pl.Callback):
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


class IncreaseAutoRegressionLengthOnPlateau(pl.Callback):
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
        max_length: float = float("inf"),
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


class CyclicTeacherForcing(pl.Callback):
    """
    Changes the teacher forcing status cyclically every cycle_in_epochs epochs.
    """

    def __init__(self, cycle_in_epochs: int, verbose: bool = False):
        """
        :param cycle_in_epochs: number of epochs after which teacher forcing is toggled
        :param verbose: if True, prints the teacher forcing status when it is changed
        """
        super().__init__()

        self.cycle_in_epochs = cycle_in_epochs
        self.verbose = verbose

    def on_train_start(self, trainer: pl.Trainer, _: Any) -> None:
        if self.verbose:
            print(f"{self.__class__.__name__}: initial teacher forcing = {trainer.teacher_forcing}")

    def on_train_epoch_start(self, trainer: pl.Trainer, _: Any) -> None:
        if trainer.current_epoch == 0:  # do not change teacher forcing at the start of training
            return

        if trainer.current_epoch % self.cycle_in_epochs == 0:
            trainer.model.teacher_forcing = not trainer.model.teacher_forcing

        if self.verbose:
            print(
                f"{self.__class__.__name__}: teacher forcing = {trainer.model.teacher_forcing}"
                f" at epoch {trainer.current_epoch}"
            )


class CombinedAutoRegressionCallback(pl.Callback):
    """
    Combined callback for auto-regression training, which changes:
        * Auto-regression length
        * Teacher forcing status
        * Learning rate

    The callback monitors certain metric and switches between three changes done to increase the training difficulty or
    reduce the learning rate, in order to keep improving the model. The order of changes of the parameters is controlled
    by callback parameters.

    :note: elements on the cycles list must be one of {"ar_length", "teacher_forcing", "learning_rate"}, but there can
           be repetitions. The order of the elements will determine the order of switches performed during the training.
    """

    def __init__(
        self,
        cycles: list[Literal["ar_length", "teacher_forcing", "learning_rate"]],
        monitor: str,
        patience: int,
        ar_length_factor: int,
        lr_factor: float,
        threshold: float = 1e-4,
        threshold_mode: Literal["abs", "rel"] = "rel",
        max_length: float = float("inf"),
        verbose: bool = False,
        reset_learning_rate: bool = False,
    ):
        """
        :param cycles: list of three strings, each must be one of {"ar_length", "teacher_forcing", "learning_rate"}
                       this order will determine the order of switches performed during the training
        :param monitor: quantity to be monitored given as key from callback_metrics dictionary of pl.Trainer
        :param patience: number of epochs with no improvement after which auto-regression length will be increased
        :param ar_length_factor: factor by which to increase auto-regression length. new_length = old_length * factor
        :param lr_factor: factor by which to decrease learning rate. new_lr = old_lr * factor
        :param threshold: threshold for measuring the new optimum, to only focus on significant changes
        :param threshold_mode: one of {"rel", "abs"}, defaults to "rel"
        :param max_length: maximum auto-regression length, defaults to no limit (infinite length)
        :param verbose: if True, prints the auto-regression length when it is changed
        """
        if any([c for c in cycles if c not in {"ar_length", "teacher_forcing", "learning_rate"}]):
            raise ValueError(
                f"{self.__class__.__name__}: cycles must have length of 3 and contain only"
                f"'ar_length', 'teacher_forcing' and 'learning_rate'!"
            )

        self.cycles = cycles
        self.monitor = monitor
        self.patience = patience
        self.ar_length_factor = ar_length_factor
        self.lr_factor = lr_factor
        self.threshold = threshold
        self.threshold_mode = threshold_mode
        self.max_length = max_length
        self.verbose = verbose
        self.reset_learning_rate = reset_learning_rate
        # placeholders and running variables
        self.current_cycle = 0
        self.best = float("inf")
        self.num_bad_epochs = 0
        self.initial_lr = None

    def is_better(self, current: float, best: float) -> bool:
        if self.threshold_mode == "rel":
            return current < best * (float(1) - self.threshold)

        else:  # self.threshold_mode == "abs":
            return current < best - self.threshold

    def detect_plateau(self, current: float) -> bool:
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            self.num_bad_epochs = 0  # reset bad epochs counter
            return True
        else:
            return False

    def switch_ar_length(self, trainer: pl.Trainer):
        new_length = trainer.datamodule.n_forward_time_steps * self.ar_length_factor

        if new_length > self.max_length:
            if self.verbose:
                print(f"{self.__class__.__name__}: maximum length reached, not increasing")
            return  # exit function is new length is greater than maximum length

        if self.verbose:
            print(f"{self.__class__.__name__}: teacher forcing = {new_length}")
        trainer.datamodule.n_forward_time_steps = new_length

    def switch_teacher_forcing(self, trainer: pl.Trainer) -> None:
        if self.verbose:
            print(f"{self.__class__.__name__}: teacher forcing = {not trainer.model.teacher_forcing}")
        trainer.model.teacher_forcing = not trainer.model.teacher_forcing

    def switch_learning_rate(self, trainer: pl.Trainer) -> None:
        if self.initial_lr is None:
            raise ValueError(f"{self.__class__.__name__}: initial_lr is None!")

        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                new_lr = param_group["lr"] * self.lr_factor
                if self.verbose:
                    print(f"{self.__class__.__name__}: learning rate = {new_lr}")
                param_group["lr"] = new_lr

    def on_train_start(self, trainer: pl.Trainer, _: Any) -> None:
        self.initial_lr = trainer.optimizers[0].param_groups[0]["lr"]

    def _reset_lr(self, trainer: pl.Trainer) -> None:
        for optimizer in trainer.optimizers:
            for param_group in optimizer.param_groups:
                param_group["lr"] = self.initial_lr

    def on_validation_epoch_end(self, trainer: pl.Trainer, _: Any) -> None:
        current = trainer.callback_metrics.get(self.monitor)
        if current is None:
            raise RuntimeError(f"{self.__class__.__name__}: metric {self.monitor} not found in callback_metrics!")
        if self.detect_plateau(current):
            if self.verbose:
                print(f"{self.__class__.__name__}: plateau detected at epoch {trainer.current_epoch}")

            switch = self.cycles[self.current_cycle]
            self.current_cycle += 1

            if switch == "ar_length":
                self.switch_ar_length(trainer)
            elif switch == "teacher_forcing":
                self.switch_teacher_forcing(trainer)
            elif switch == "learning_rate":
                self.switch_learning_rate(trainer)

            if self.current_cycle == len(self.cycles):
                if self.verbose:
                    print(f"{self.__class__.__name__}: auto-regression callback cycle completed!")
                if self.reset_learning_rate:
                    self._reset_lr(trainer)

                self.current_cycle = 0
