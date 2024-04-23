import lightning.pytorch as pl
import pytest
from torch.utils.data import DataLoader

from pydentification.training.lightning.callbacks import CyclicTeacherForcing

from .mocks import RandDataset, StepAheadModule, ZeroLossPredictionTrainer


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
    trainable_module = ZeroLossPredictionTrainer(
        module=StepAheadModule(), teacher_forcing=True, log_attrs={"teacher_forcing"}
    )
    dataloader = DataLoader(RandDataset(size=10, shape=(10, 1)), batch_size=1, shuffle=False)

    trainer = pl.Trainer(
        accelerator="cpu",  # only CPU for unit-tests
        max_epochs=max_epochs,
        callbacks=[CyclicTeacherForcing(cycle_in_epochs=cycle_in_epochs)],
    )

    trainer.fit(trainable_module, dataloader)
    # each epoch teacher forcing is logged with key to distinguish epochs
    teacher_forcing_history = {
        key: bool(value) for key, value in trainer.logged_metrics.items() if key.startswith("teacher_forcing")
    }

    assert list(teacher_forcing_history.values()) == expected
