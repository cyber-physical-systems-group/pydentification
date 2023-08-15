from typing import Optional, Type

import lightning.pytorch as pl
import torch

from .mixins import LightningRegressionMixin


def to_lightning_module(
    module: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    lr_scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
    loss: Optional[torch.nn.Module] = None,
    meta: Type = LightningRegressionMixin,
) -> pl.LightningModule:
    """
    Wrapper converting pytorch module into Lightning module

    This allows to separate the logic of the model from the logic of the training, where most of the models can be
    trained using the default metaclass `LightningRegressionMixin`,which implements standard regression training.

    :param module: pytorch module to be wrapped
    :param optimizer: initialized optimizer to be used for training
    :param lr_scheduler: initialized learning rate scheduler to be used for training
    :param loss: loss functional as callable module of function
    :param meta: type of the metaclass, which needs to implement training interface for lightning
                 and it needs to be model agnostic, if possible
    """

    class WrappedModel(meta):
        def __init__(self, module_, optimizer_, lr_scheduler_, loss_):
            super().__init__(module_, optimizer_, lr_scheduler_, loss_)  # type: ignore

    return WrappedModel(module, optimizer, lr_scheduler, loss)
