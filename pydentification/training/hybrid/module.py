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

    def forward(self, y_true: Tensor, y_pred: Tensor, lower: Tensor | None = None, upper: Tensor | None = None) -> float:
        loss = F.mse_loss(y_pred, y_true, reduction="mean")

        # if gamma is given as 0 or if bounds are not given,
        # no penalty is applied and BoundedMSELoss is equivalent to MSE, for example when used as validation loss
        if self.gamma == 0 or lower is None or upper is None:
            return loss

        penalty = torch.where(
            (y_pred < lower) | (y_pred > upper),  # find predictions outside of bounds
            torch.min(torch.abs(y_pred + upper), torch.abs(y_pred - lower)),  # calculate distance to closer bound
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
        """
        :param network: initialized neural network to be wrapped by HybridBoundedSimulationTrainingModule
        :param estimator: non-parametric estimator to be used for theoretical bounds
        :param optimizer: initialized optimizer to be used for training
        :param bound_during_training: flag to enable bounding during training, defaults to False
        :param bound_crossing_penalty: penalty factor for crossing bounds, see: BoundedMSELoss, defaults to 0.0
        :param lr_scheduler: initialized learning rate scheduler to be used for training, defaults to None
        """
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

    def forward(self, x: Tensor, return_nonparametric: bool = False) -> Tensor:
        nonparametric_predictions, bounds = self.estimator(x)
        # bounds are returned as distance from nonparametric predictions
        upper_bound = nonparametric_predictions + bounds
        lower_bound = nonparametric_predictions - bounds

        predictions = self.network(x)

        if self.bound_during_training:
            predictions = bounded_linear_unit(predictions, lower=lower_bound, upper=upper_bound)

        if return_nonparametric:
            return predictions, nonparametric_predictions, lower_bound, upper_bound

        return predictions, lower_bound, upper_bound

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        y_hat, lower_bound, upper_bound = self.forward(x)  # type: ignore
        loss = self.loss(y_hat, y, lower_bound, upper_bound)  # type: ignore
        self.log("training/train_loss", loss)

        return loss

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        x, y = batch
        # bounds are not used for validation
        y_hat, _, _ = self.forward(x)  # type: ignore
        loss = self.loss(y_hat, y)  # type: ignore
        self.log("training/validation_loss", loss)

        return loss

    def on_train_epoch_end(self):
        self.log("training/lr", self.trainer.optimizers[0].param_groups[0]["lr"])

    def configure_optimizers(self) -> dict[str, Any]:
        config = {"optimizer": self.optimizer, "lr_scheduler": self.lr_scheduler, "monitor": "training/validation_loss"}
        return {key: value for key, value in config.items() if value is not None}  # remove None values
