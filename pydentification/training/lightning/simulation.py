from typing import Any, Optional

import lightning.pytorch as pl
import torch


class LightningSimulationTrainingModule(pl.LightningModule):
    """
    Lightning module for training n-step-ahead simulation model
    Refer to lightning docs for more details: https://lightning.ai/docs/pytorch/stable/starter/converting.html
    """

    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer],
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        loss: Optional[torch.nn.Module] = torch.nn.functional.mse_loss,
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

        self.save_hyperparameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.module(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.module(x)  # type: ignore
        loss = self.loss(y_hat, y)
        self.log("training/train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self.module(x)  # type: ignore
        loss = self.loss(y_hat, y)
        self.log("training/validation_loss", loss)

        return loss

    def on_train_epoch_end(self):
        self.log("training/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, _: int = 0) -> torch.Tensor:
        """
        Warning: this does not work when using distributed training, recommended solution is to predict on CPU or
        use different Lightning wrapper, see: https://github.com/Lightning-AI/lightning/issues/10618
        """
        x, _ = batch
        return self.module(x)  # type: ignore

    def configure_optimizers(self) -> dict[str, Any]:
        return {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "training/validation_loss"}
