from typing import Any

import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn import Module
from torch.nn import functional as F

from pydentification.models.modules.activations import bounded_linear_unit

from .estimator import KernelRegression


class BoundedMSELoss(Module):
    """MSE loss with penalty for crossing bounds of the nonparametric estimator"""

    def __init__(self, gamma: float):
        """
        :param gamma: penalty factor for crossing bounds, defaults to 0.0
        """
        self.gamma = gamma

        super(BoundedMSELoss, self).__init__()

    def forward(self, y_true: Tensor, y_pred: Tensor, bounds: Tensor) -> float:
        loss = F.mse_loss(y_pred, y_true, reduction="mean")

        # if gamma is given as 0, no penalty is applied and BoundedMSELoss is equivalent to MSE
        # or if bounds are not given, no penalty is applied, for example when used as validation loss
        if self.gamma == 0 or bounds is None:
            return loss

        penalty = torch.where(
            (y_pred < -bounds) | (y_pred > bounds),  # find predictions outside of bounds
            torch.min(torch.abs(y_pred + bounds), torch.abs(y_pred - bounds)),  # calculate distance to closer bound
            torch.tensor(0.0, device=y_pred.device),  # zero-fill for predictions inside bounds
        )

        # returns loss as sum of MSE and cumulated penalty for crossing bounds with gamma factor
        return loss + self.gamma * torch.sum(penalty)


class HybridBoundedSimulationTrainingModule(pl.LightningModule):
    """
    This class contains training module for neural network to identify nonlinear dynamical systems or static nonlinear
    functions with guarantees by using bounded activation incorporating theoretical bounds from the kernel regression
    estimator. The approach is limited to finite memory single-input single-output dynamical systems,
    which can be converted to static multiple-input single-output systems by using delay-line.

    Bounds are computed using kernel regression working with the same data, but we are able to guarantees of the
    estimation, which are used to activate a network during and after training, in order to ensure that the predictions
    are never outside of those theoretical bounds.

    Bounds can be also used as penalty during training, which is implemented in this class using `BoundedMSELoss`.
    """

    def __init__(
        self,
        network: Module,
        estimator: KernelRegression,
        optimizer: torch.optim.Optimizer,
        bound_during_training: bool = False,
        bound_crossing_penalty: float = 0.0,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    ):
        super().__init__()

        self.network = network
        self.estimator = estimator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler

        self.bound_during_training = bound_during_training
        self.loss = BoundedMSELoss(gamma=bound_crossing_penalty)

        self.save_hyperparameters()

    @classmethod
    def from_pretrained(cls, trained_network: Module, **kwargs):
        """
        Shortcut for using module with pretrained network. Calling this method is equivalent to passing the trained
        network directly to `__init__`, but the classmethod can be useful for stating the user intention.
        """
        return cls(network=trained_network, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        nonparametric_predictions, bounds = self.estimator(x)
        # bounds are returned as distance from nonparametric predictions
        bounds = nonparametric_predictions + bounds
        predictions = self.network(x)

        if self.bound_during_training:
            predictions = bounded_linear_unit(predictions, bounds)

        return predictions, bounds

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        y_hat, bounds = self.forward(x)  # type: ignore
        loss = self.loss(y_hat, y, bounds)  # type: ignore
        self.log("training/train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        # bounds are not used for validation
        y_hat, _ = self.forward(x)  # type: ignore
        loss = self.loss(y_hat, y, bounds=None)  # type: ignore
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
        return self.forward(x)  # type: ignore

    def configure_optimizers(self) -> dict[str, Any]:
        config = {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "training/validation_loss"}
        return {key: value for key, value in config.items() if value is not None}  # remove None values
