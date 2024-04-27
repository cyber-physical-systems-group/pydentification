import lightning.pytorch as pl
import pytest
from torch.utils.data import DataLoader

from pydentification.training.lightning import callbacks

from .mocks import FunctionLossPredictionTrainer, RandDataset, StepAheadModule, ZeroLossPredictionTrainer


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
@pytest.mark.skip
def test_cyclic_teacher_forcing(cycle_in_epochs: int, max_epochs: int, expected: list[bool]):
    trainable_module = ZeroLossPredictionTrainer(module=StepAheadModule(), teacher_forcing=True)
    dataloader = DataLoader(RandDataset(size=10, shape=(10, 1)), batch_size=1, shuffle=False)

    trainer = pl.Trainer(
        accelerator="cpu",  # only CPU for unit-tests
        max_epochs=max_epochs,
        callbacks=[callbacks.CyclicTeacherForcing(cycle_in_epochs=cycle_in_epochs)],
    )

    trainer.fit(trainable_module, dataloader)
    # each epoch teacher forcing is logged with key to distinguish epochs
    metrics = trainer.logged_metrics.items()
    teacher_forcing_history = {key: bool(value) for key, value in metrics if key.startswith("teacher_forcing")}

    assert list(teacher_forcing_history.values()) == expected


@pytest.mark.parametrize(
    ["mock_loss_fn", "max_epochs", "patience", "factor", "max_length", "expected_lengths"],
    (
        # constant loss so trainer sees no improvement
        # 5 epoch patience increases the length by 2 at 5 epoch and again at 10 epoch
        # length is logged for each epoch at the end, so after changing with the callback
        (lambda x: float(0), 10, 5, 2, float("inf"), [1, 1, 1, 1, 2, 2, 2, 2, 2, 4]),
        # linearly decreasing loss, so trainer sees improvement and keeps the length
        (lambda x: float(10 - x), 10, 5, 2, float("inf"), [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
        # linearly increasing loss, length is increased every epoch (patience is 1)
        (lambda x: float(x), 10, 1, 2, float("inf"), [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]),
        # loss is dropping every second epoch
        (lambda x: 10 - x - (x % 2), 10, 1, 2, float("inf"), [2, 2, 4, 4, 8, 8, 16, 16, 32, 32]),
        # max length is reached in 5 epoch
        (lambda x: float(x), 10, 1, 2, 32, [2, 4, 8, 16, 32, 32, 32, 32, 32, 32]),
    ),
)
def test_increase_autoregression_length_on_plateau(
    mock_loss_fn,
    max_epochs: int,
    patience: int,
    factor: int,
    max_length: float | int,
    expected_lengths: list[int],
    random_prediction_datamodule,
):
    trainable_module = FunctionLossPredictionTrainer(StepAheadModule(), loss_fn=mock_loss_fn)
    callback = callbacks.IncreaseAutoRegressionLengthOnPlateau(
        monitor="val_loss",
        patience=patience,
        factor=factor,
        max_length=max_length,
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=max_epochs,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[callback],
    )

    random_prediction_datamodule.n_forward_time_steps = 1  # set to 1 before fit, the datamodule is shared between tests
    trainer.fit(trainable_module, datamodule=random_prediction_datamodule)

    metrics = trainer.logged_metrics.items()
    ar_length_history = [int(value) for key, value in metrics if key.startswith("n_forward_time_steps")]
    assert ar_length_history == expected_lengths
