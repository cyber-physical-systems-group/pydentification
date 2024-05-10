from typing import Any, Callable

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn import Module


class StepAheadModule(Module):
    """
    Mocked torch module for testing functionalities with one-step ahead training.
    It takes any input and always returns one step of zeros with the same dimension as inputs.

    The module has single parameter, so tests of training methods computing gradients can be performed.
    Gradients are zero all the time, since the output is always constant.
    """

    def __init__(self, teacher_forcing: bool = False, full_residual_connection: bool = False):
        super(StepAheadModule, self).__init__()

        self.teacher_forcing = teacher_forcing
        self.full_residual_connection = full_residual_connection

        self.parameter = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.parameter * torch.zeros_like(x[:, 0, :])


class MockLightningTrainer:
    """
    Mock of Lightning Trainer for testing purposes. It only calls required functions in expected order without
    implementing the training logic, to allow testing callbacks etc. without long runtime of tests.

    :note: actual pl.Trainer might have different order of calling callbacks and reloading datamodule and model.
           the properties changed by callbacks might be off by 1 epoch in assert, depending on when they are logged.
    """

    def __init__(
        self, loss_fn: Callable[[int], Tensor], callbacks: list[pl.Callback], max_epochs: int, *args, **kwargs
    ):
        """
        :param loss_fn: function returning loss value for given epoch
        :param callbacks: list of callbacks to be used during training
        :param max_epochs: number of epochs to train
        """
        self.loss_fn = loss_fn
        self.max_epochs = max_epochs
        self.callbacks = callbacks
        # placeholders
        self.model = None
        self.datamodule = None
        self.optimizers = None
        # running variables
        self.current_epoch = 0
        self.callback_metrics = {}
        self.logged_metrics = []

    def store(self, datamodule: pl.LightningDataModule, model: pl.LightningModule) -> dict[str, Any]:
        """Store properties modified by callbacks to check them in tests"""
        time_steps = datamodule.n_forward_time_steps
        epoch_metrics = {"n_forward_time_steps": time_steps, "teacher_forcing": model.teacher_forcing}

        for index, optimizer in enumerate(self.optimizers):
            for group_index, param_group in enumerate(optimizer.param_groups):
                # log learning rate for each epoch for each optimizer and its parameter groups
                epoch_metrics[f"learning_rate_optimizer_{index}_{group_index}"] = param_group["lr"]

        return epoch_metrics

    def fit(self, model: pl.LightningModule, datamodule: pl.LightningDataModule):
        """
        Mocked fit method, calling callbacks in expected order and storing metrics for testing.
        note: this does not run actual training or Any Tensor computations, only calls callbacks in expected order.
        """
        # store model, optimizers and datamodule for callbacks
        self.model = model
        self.optimizers = [torch.optim.Adam(model.parameters(), lr=1)]
        self.datamodule = datamodule

        for callback in self.callbacks:
            callback.on_train_start(self, self.model)

        for epoch in range(self.max_epochs):
            self.current_epoch = epoch  # set current epoch for callbacks

            for callback in self.callbacks:
                callback.on_train_epoch_start(self, self.model)
                callback.on_validation_epoch_start(self, self.model)

            loss_value = self.loss_fn(self.current_epoch)
            loss = torch.autograd.Variable(torch.Tensor([loss_value]), requires_grad=True)
            self.callback_metrics["val_loss"] = loss

            for callback in self.callbacks:
                callback.on_train_epoch_end(self, self.model)
                callback.on_validation_epoch_end(self, self.model)

            self.logged_metrics.append(self.store(self.datamodule, self.model))

        for callback in self.callbacks:
            callback.on_train_end(self, self.model)
