from typing import Any, Callable

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LightningSimulationTrainingModule(pl.LightningModule):
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

    def forward(self, x: Tensor) -> Tensor:
        return self.module(x)

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.module(x)  # type: ignore
        loss = self.loss(y_hat, y)
        self.log("trainer/train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat = self.module(x)  # type: ignore
        loss = self.loss(y_hat, y)
        self.log("trainer/validation_loss", loss)

        return loss

    def on_train_epoch_end(self):
        self.log("trainer/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def predict_step(self, batch: tuple[Tensor, Tensor], batch_idx: int, _: int = 0) -> Tensor:
        """
        Warning: this does not work when using distributed training, recommended solution is to predict on CPU or
        use different Lightning wrapper, see: https://github.com/Lightning-AI/lightning/issues/10618
        """
        x, _ = batch
        return self.module(x)  # type: ignore

    def configure_optimizers(self) -> dict[str, Any]:
        config = {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "trainer/validation_loss"}
        return {key: value for key, value in config.items() if value is not None}  # remove None values
