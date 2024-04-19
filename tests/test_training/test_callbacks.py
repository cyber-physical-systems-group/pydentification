import lightning.pytorch as pl
import pytest
import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader

from pydentification.training.lightning.callbacks import CyclicTeacherForcing

from .mocks import RandDataset, StepAheadModule


class MockPredictionTrainer(pl.LightningModule):
    def __init__(
        self,
        module: Module,
        teacher_forcing: bool = False,
    ):
        super().__init__()

        self.module = module
        self.teacher_forcing = teacher_forcing

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
        # log teacher forcing status at the end of each epoch with key to distinguish epochs
        self.log_dict({f"teacher_forcing_at_{self.current_epoch}": self.teacher_forcing}, on_epoch=True, on_step=False)


@pytest.mark.parametrize(
    ["cycle_in_epochs", "max_epochs", "expected"],
    (
        # cycle every epoch
        (1, 10, [True, False, True, False, True, False, True, False, True, False]),
        # cycle every 2 epochs
        (2, 10, [True, True, False, False, True, True, False, False, True, True]),
        # cycle every 3 epochs
        (3, 10, [True, True, True, False, False, False, True, True, True, False]),
        # cycle for entire training
        (5, 5, [True, True, True, True, True]),
    ),
)
def test_cyclic_teacher_forcing(cycle_in_epochs: int, max_epochs: int, expected: list[bool]):
    # start from teacher forcing enabled
    trainable_module = MockPredictionTrainer(module=StepAheadModule(), teacher_forcing=True)
    dataloader = DataLoader(RandDataset(size=10, shape=(10, 1)), batch_size=1, shuffle=False)

    trainer = pl.Trainer(
        accelerator="cpu",  # only CPU for unit-tests
        max_epochs=max_epochs,
        callbacks=[CyclicTeacherForcing(cycle_in_epochs=cycle_in_epochs)],
    )

    trainer.fit(trainable_module, dataloader)
    # each epoch teacher forcing is logged with key to distinguish epochs
    teacher_forcing_history = [bool(v.item()) for v in trainer.logged_metrics.values()]
    assert teacher_forcing_history == expected
