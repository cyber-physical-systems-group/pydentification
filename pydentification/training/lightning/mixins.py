from typing import Optional

import lightning.pytorch as pl
import torch


class LightningRegressionMixin(pl.LightningModule):
    """
    Default Lightning wrapper implementing standard regression training procedure used for most models in this use case

    It needs to implement following lightning methods:
        * forward
        * training_step
        * validation_step
        * test_step
        * configure_optimizers
        * predict_step

    Refer to lightning docs for more details: https://lightning.ai/docs/pytorch/stable/starter/converting.html
    """

    def __init__(
        self,
        module: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        loss: Optional[torch.nn.Module] = None,
    ):
        """
        :param module: initialized module to be wrapped
        :param optimizer: initialized optimizer to be used for training,
                          if None Adam with default settings is used
        :param lr_scheduler: initialized learning rate scheduler to be used for training,
                             if None given no LR schedule is used
        :param loss: loss functional as callable module of function, if None MSE is used
        """
        super().__init__()

        self.wrapped_module = module
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss = loss or torch.nn.functional.mse_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.wrapped_module(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)  # type: ignore
        loss = self.loss(y_hat, y)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)  # type: ignore
        loss = self.loss(y_hat, y)

        return loss

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat = self(x)  # type: ignore
        loss = self.loss(y_hat, y)

        return loss

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int, _: int = 0) -> torch.Tensor:
        """
        Warning: this does not work when using distributed training, recommended solution is to predict on CPU or
        use different Lightning wrapper, see: https://github.com/Lightning-AI/lightning/issues/10618
        """
        x, _ = batch
        return self(x)  # type: ignore

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.optimizer is None:
            optimizer = torch.optim.Adam(self.parameters())  # use Adam by default
        else:
            optimizer = self.optimizer

        if self.lr_scheduler is None:
            return optimizer
        return {"optimizer": optimizer, "lr_scheduler": self.lr_scheduler}
