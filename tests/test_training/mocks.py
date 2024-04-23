import lightning.pytorch as pl
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset


class StepAheadModule(Module):
    """
    Mocked torch module for testing functionalities with one-step ahead training.
    It takes any input and always returns one step of zeros with the same dimension as inputs.

    The module has single parameter, so tests of training methods computing gradients can be performed.
    Gradients are zero all the time, since the output is always constant.
    """

    def __init__(self):
        super(StepAheadModule, self).__init__()

        self.parameter = torch.nn.Parameter(torch.rand(1), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return self.parameter * torch.zeros_like(x[:, 0, :])


class RandDataset(Dataset):
    """
    Random torch dataset returning given number of elements each with given shape
    """

    def __init__(self, size: int, shape: tuple):
        self.size = size
        self.shape = shape

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, _) -> Tensor:
        return torch.rand(self.shape)


class ZeroLossPredictionTrainer(pl.LightningModule):
    """
    Mock of LightningModule for training n-step-ahead simulation model, which has zero loss all the time.
    It is used to test training methods and callbacks without need of real loss computation.
    """

    def __init__(
        self,
        module: Module,
        teacher_forcing: bool = False,
        full_residual_connection: bool = False,
        log_attrs: set[str] = frozenset(),
    ):
        super().__init__()

        self.module = module

        self.teacher_forcing = teacher_forcing
        self.full_residual_connection = full_residual_connection

        self.log_attrs = log_attrs

    def configure_optimizers(self) -> dict:
        return {"optimizer": torch.optim.Adam(self.module.parameters())}

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # return 0 loss as torch.Variable with gradient to align with required interface
        return torch.autograd.Variable(torch.Tensor([float(0)]), requires_grad=True)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # return 0 loss as torch.Variable with gradient to align with required interface
        return torch.autograd.Variable(torch.Tensor([float(0)]), requires_grad=True)

    def predict_step(self, batch: tuple[Tensor, Tensor]):
        x, y = batch
        return self.module(x)

    def on_train_epoch_end(self) -> None:
        """
        Log attributes given in __init__ to the logger after each epoch.
        Each key is the epoch number and value is dict of attributes and their values after the epoch.
        """
        for attr in self.log_attrs:
            self.log_dict({f"{attr}_at_{self.current_epoch}": getattr(self, attr)}, on_epoch=True, on_step=False)


class ZeroLossPredictionTrainer(pl.LightningModule):
    """
    Mock of LightningModule for training n-step-ahead simulation model, which has zero loss all the time.
    It is used to test training methods and callbacks without need of real loss computation.
    """

    def __init__(
        self,
        module: Module,
        teacher_forcing: bool = False,
        full_residual_connection: bool = False,
        log_attrs: set[str] = frozenset(),
    ):
        super().__init__()

        self.module = module

        self.teacher_forcing = teacher_forcing
        self.full_residual_connection = full_residual_connection

        self.log_attrs = log_attrs

    def configure_optimizers(self) -> dict:
        return {"optimizer": torch.optim.Adam(self.module.parameters())}

    def training_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # return 0 loss as torch.Variable with gradient to align with required interface
        return torch.autograd.Variable(torch.Tensor([float(0)]), requires_grad=True)

    def validation_step(self, batch: tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # return 0 loss as torch.Variable with gradient to align with required interface
        return torch.autograd.Variable(torch.Tensor([float(0)]), requires_grad=True)

    def predict_step(self, batch: tuple[Tensor, Tensor]):
        x, y = batch
        return self.module(x)

    def on_train_epoch_end(self) -> None:
        """
        Log attributes given in __init__ to the logger after each epoch.
        Each key is the epoch number and value is dict of attributes and their values after the epoch.
        """
        for attr in self.log_attrs:
            self.log_dict({f"{attr}_at_{self.current_epoch}": getattr(self, attr)}, on_epoch=True, on_step=False)