from typing import Literal

import lightning.pytorch as pl


def is_better(current: float, best: float, threshold: float, threshold_mode: Literal["abs", "rel"]) -> bool:
    """
    Check if current value is better than the best value. Used for plateau detection.

    :param current: current metric value to compare.
    :param best: best metric value to compare.
    :param threshold: threshold for improvement.
    :param threshold_mode: how to interpret the threshold: "abs" is absolute or "rel" is relative,
                           which means that the threshold is a fraction of the best value.
    """
    if threshold_mode == "rel":
        return current < best * (float(1) - threshold)

    else:  # threshold_mode == "abs":
        return current < best - threshold


def switch_autoregression_length(
    trainer: pl.Trainer, target_length: int, max_length: float = float("inf"), name: str | None = None
):
    """
    Switches the autoregression length in the model.

    :param trainer: pl.Trainer instance, its datamodule will be modified (needs to be PredictionDataModule).
    :param target_length: new autoregression length, must be int and >= 1.
    :param max_length: maximum autoregression length, if new length is greater than this, it will not be increased.
    :param name: name of the function for logging.
    """
    name = name or "functional.switch_autoregression_length"

    if target_length > max_length:
        print(f"{name}: maximum length reached, not increasing")
        return  # exit function is new length is greater than maximum length

    if not isinstance(target_length, int) or target_length < 1:
        raise ValueError(f"{name}: new_length must be int and >= 1, got {target_length}")

    print(f"{name}: teacher forcing = {target_length}")
    trainer.datamodule.n_forward_time_steps = target_length


def switch_teacher_forcing(trainer: pl.Trainer, name: str | None = None):
    """
    Switches the teacher forcing mode in the model.

    :param trainer: pl.Trainer instance in training mode, its model will be modified.
    :param name: name of the function for logging.
    """
    name = name or "functional.switch_teacher_forcing"

    if not hasattr(trainer.model, "teacher_forcing"):
        raise AttributeError(f"{name}: trainer has no attribute 'teacher_forcing'!")

    print(f"{name}: teacher forcing = {not trainer.model.teacher_forcing}")
    trainer.model.teacher_forcing = not trainer.model.teacher_forcing


def switch_learning_rate(trainer: pl.Trainer, lr_factor: float, name: str | None = None):
    """
    Multiplies all learning rates in all optimizer and parameter groups by the factor.

    :param trainer: pl.Trainer instance in training mode, its optimizer(s) will be modified.
    :param lr_factor: factor by which to multiply all learning rates, should follow: 0 > LR_FACTOR > 1
    :param name: name of the function for logging.
    """
    name = name or "functional.switch_learning_rate"

    for optimizer in trainer.optimizers:
        for param_group in optimizer.param_groups:
            new_lr = param_group["lr"] * lr_factor
            print(f"{name}: learning rate = {new_lr}")
            param_group["lr"] = new_lr


def reset_lr(trainer: pl.Trainer, initial_lrs: list[list[float]]):
    """
    Resets all learning rates in all optimizer and parameter groups to the initial values.

    :param trainer: pl.Trainer instance in training mode, its optimizer(s) will be modified.
    :param initial_lrs: list of lists of initial learning rates for each optimizer and parameter group.
    """
    for optimizer, initial_lr in zip(trainer.optimizers, initial_lrs, strict=True):
        for param_group, param_group_lr in zip(optimizer.param_groups, initial_lr, strict=True):
            param_group["lr"] = param_group_lr
