from typing import Literal

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


@pytest.mark.parametrize(
    [
        "cycles",
        "reset_learning_rate",
        "expected_lengths",
        "expected_learning_rates",
        "expected_teacher_forcing",
    ],
    (
        (
            # two cycles in the callback - one in 5 epoch and one in 10 epoch, loss does not change and patience is 5
            # first cycle increases the length by 2, second cycle decreases the learning rate to 0.1
            ["ar_length", "learning_rate"],
            False,  # reset_learning_rate
            # auto-regression length is increased by 2 at 5 epoch
            [1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8],
            [1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.001],
            [True] * 12,  # teacher forcing is not changed, since it is not in cycles
        ),
        (
            ["ar_length", "learning_rate"],
            True,  # reset_learning_rate
            # callback is called each second epoch
            # auto-regression length is increased at 2nd, 6th and 10th epoch
            [1, 2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8],
            # learning rate is reset at the end of each cycle, so it does not change
            [float(1)] * 12,
            [True] * 12,  # teacher forcing is not changed, since it is not in cycles
        ),
        (
            ["learning_rate", "ar_length"],
            True,  # reset_learning_rate
            # callback is called each second epoch
            # auto-regression length is increased at 2nd, 6th and 10th epoch
            [1, 1, 1, 2, 2, 2, 2, 4, 4, 4, 4, 8],
            # learning rate is reset at the end of each cycle, so it does not change
            [1, 0.1, 0.1, 1, 1, 0.1, 0.1, 1, 1, 0.1, 0.1, 1],
            [True] * 12,  # teacher forcing is not changed, since it is not in cycles
        ),
        (
            ["learning_rate", "learning_rate", "learning_rate", "ar_length"],
            True,  # reset_learning_rate
            # auto-regression length is increased at 8th epoch
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            # learning rate is decreases every second epoch and is reset at epoch 8 - at the end of the cycle
            [1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 1, 1, 0.1, 0.1, 0.01],
            [True] * 12,  # teacher forcing is not changed, since it is not in cycles
        ),
        (
            ["learning_rate", "learning_rate", "learning_rate", "ar_length", "ar_length"],
            True,  # reset_learning_rate
            # auto-regression length is increased at 8th and 10-th epoch
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 4, 4, 4],
            # learning rate is decreases until epoch 8 when cycle goes to increasing AR length twice
            # it is reset at epoch 10 - at the end of the cycle
            [1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 1, 1, 0.1],
            [True] * 12,  # teacher forcing is not changed, since it is not in cycles
        ),
        (
            ["learning_rate", "ar_length", "teacher_forcing"],
            False,  # reset_learning_rate
            # auto-regression length is increased at 8th and 10-th epoch
            [1, 1, 1, 2, 2, 2, 2, 2, 2, 4, 4, 4],
            # learning rate is decreases until epoch 8 when cycle goes to increasing AR length twice
            # it is reset at epoch 10 - at the end of the cycle
            [1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.01, 0.01, 0.01, 0.01, 0.01],
            # teacher forcing is changed at 6th and 12th epoch
            [True, True, True, True, True, False, False, False, False, False, False, True],
        ),
        (
            ["learning_rate", "learning_rate", "learning_rate", "ar_length", "teacher_forcing"],
            True,  # reset_learning_rate
            [1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            [1, 0.1, 0.1, 0.01, 0.01, 0.001, 0.001, 0.001, 0.001, 1, 1, 0.1],
            [True, True, True, True, True, True, True, True, True, False, False, False],
        ),
    ),
)
def test_combined_autoregression_callback_cycles(
    cycles: list[Literal["ar_length", "teacher_forcing", "learning_rate"]],
    reset_learning_rate: bool,
    expected_lengths: list[int],
    expected_learning_rates: list[float],
    expected_teacher_forcing: list[bool],
    random_prediction_datamodule,
):
    trainable_module = FunctionLossPredictionTrainer(
        StepAheadModule(),
        loss_fn=lambda x: float(0),
        teacher_forcing=True,
    )

    callback = callbacks.CombinedAutoRegressionCallback(
        cycles=cycles,
        monitor="val_loss",
        patience=2,
        ar_length_factor=2,
        lr_factor=0.1,
        reset_learning_rate=reset_learning_rate,
        max_length=float("inf"),
        verbose=True,
    )

    trainer = pl.Trainer(
        accelerator="cpu",
        max_epochs=12,
        reload_dataloaders_every_n_epochs=1,
        callbacks=[callback],
    )

    random_prediction_datamodule.n_forward_time_steps = 1  # set to 1 before fit, the datamodule is shared between tests
    trainer.fit(trainable_module, datamodule=random_prediction_datamodule)

    metrics = trainer.logged_metrics.items()
    ar_length_history = [int(value) for key, value in metrics if key.startswith("n_forward_time_steps")]
    learning_rate_history = [pytest.approx(float(value)) for key, value in metrics if key.startswith("learning_rate")]
    teacher_forcing_history = [bool(value) for key, value in metrics if key.startswith("teacher_forcing")]

    assert ar_length_history == expected_lengths
    assert learning_rate_history == expected_learning_rates
    assert teacher_forcing_history == expected_teacher_forcing
