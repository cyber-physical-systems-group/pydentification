from typing import Any, Callable

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LightningPredictionTrainingModule(pl.LightningModule):
    """
    Lightning module for training n-step-ahead simulation model
    Refer to lightning docs for more details: https://lightning.ai/docs/pytorch/stable/starter/converting.html
    """

    def __init__(
        self,
        module: Module,
        optimizer: Optimizer,
        lr_scheduler: LRScheduler | None = None,
        loss: Module | Callable = torch.nn.functional.mse_loss,
        teacher_forcing: bool = False,
        full_residual_connection: bool = False,
    ):
        """
        :param module: initialized module to be wrapped
        :param optimizer: initialized optimizer to be used for training
        :param lr_scheduler: initialized learning rate scheduler to be used for training
        :param loss: loss functional as callable module of function, defaults to MSE
        """
        super().__init__()

        self.module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss

        self.teacher_forcing = teacher_forcing
        self.full_residual_connection = full_residual_connection

        self.save_hyperparameters()

    def forward(self, x: Tensor) -> Tensor:
        y_hat = self.module(x)

        if self.full_residual_connection:
            y_hat = y_hat + x[:, -1, :]  # model only predicts delta

        return y_hat

    def unroll_forward(self, batch: tuple[Tensor, Tensor], teacher_forcing: bool) -> Tensor:
        x, y = batch
        predictions = torch.empty_like(y)

        for step in range(predictions.shape[1]):  # iterate over time steps
            # account for auto-regression longer then initial input
            ar_start_idx = max(0, step - x.shape[1])
            if teacher_forcing:
                # concat inputs with targets in teacher forcing
                step_inputs = torch.cat([x[:, step:, :], y[:, ar_start_idx:step, :]], dim=1)
            else:
                step_inputs = torch.cat([x[:, step:, :], predictions[:, ar_start_idx:step, :]], dim=1)

            y_hat = self.module(step_inputs)
            predictions[:, step, :] = y_hat[:, 0, :]

        return predictions

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch

        predictions = self.unroll_forward(batch, self.teacher_forcing)

        loss = self.loss(predictions, y)
        self.log("training/train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch

        predictions = self.unroll_forward(batch, teacher_forcing=False)  # never use teacher forcing during validation

        loss = self.loss(predictions, y)
        self.log("training/validation_loss", loss)

        return loss

    def on_train_epoch_end(self):
        self.log("training/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, dataloader_idx: int = 0) -> Tensor:
        """Requires using batch of training inputs and targets to know the number of time steps to predict"""
        return self.unroll_forward(batch, teacher_forcing=False)  # never use teacher forcing during prediction

    def configure_optimizers(self) -> dict[str, Any]:
        config = {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "training/validation_loss"}
        return {key: value for key, value in config.items() if value is not None}  # remove None values
